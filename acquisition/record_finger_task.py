"""
Enregistrement EEG (Mentalab Explore_8547_ExG, 32 canaux, via LSL) pour une
tache de detection de mouvement des doigts.

Protocole : 4 conditions (thumb + index / majeur / annulaire / auriculaire),
main droite, N_REPS_PER_FINGER repetitions chacune (20 par defaut), soit
4 x 20 = 80 essais.

Design :
  - Le sujet appuie, AVEC le doigt indique par la consigne visuelle, la touche
    clavier correspondante (convention dactylographie main droite) :
        Index - 1       -> H
        Majeur - 2      -> J
        Annulaire - 3   -> K
        Auriculaire - 4 -> L 
    Cela donne un timestamp precis de la pression sans materiel supplementaire.
    Si le clavier n'est pas en QWERTY, adaptez FINGER_KEYS ci-dessous (les
    valeurs doivent etre des noms de "keysym" Tkinter).

  - L'EEG est enregistre en continu dans un thread separe pendant toute la
    duree de la tache (pas d'interruption entre essais).

  - Un flux LSL de marqueurs ("FingerTaskMarkers") est pousse en parallele de
    l'EEG. Les deux partagent la meme horloge LSL (local_clock), donc
    l'alignement des evenements sur l'EEG est precis meme si l'interface
    graphique a un peu de latence.

  - A la fin : sauvegarde d'un fichier .fif (raw + annotations MNE) et d'un
    fichier .csv detaillant chaque essai (doigt attendu, doigt presse, temps
    de reaction, erreur eventuelle).

Dependances : pylsl, mne, numpy (tkinter est fourni avec Python).
"""

import os
import csv
import time
import random
import threading

import numpy as np
import mne
import tkinter as tk

from pylsl import StreamOutlet, StreamInfo, local_clock
from Explore_EEG import Explore_EEG

# ---------------------------------------------------------------------------
# CONFIGURATION - a adapter selon vos besoins
# ---------------------------------------------------------------------------
SUBJECT = "001"
SESSION = 2
RUN = 1

N_REPS_PER_FINGER = 4        # nb de repetitions par doigt
RANDOMIZE_ORDER = True        # False = enregistrement en blocs (20 essais du meme doigt, puis le suivant)
TRIALS_PER_BREAK = 20         # pause longue toutes les N essais (mettre >= len(trials) pour desactiver)

FIXATION_MIN, FIXATION_MAX = 1.0, 1.5   # jitter de la croix de fixation (s), evite l'anticipation
MAX_RESPONSE_TIME = 4.0                 # delai max pour repondre avant "trop lent" (s)
POST_PRESS_FEEDBACK = 0.3               # duree d'affichage du feedback (s)
INTER_TRIAL_REST = 1.5                  # repos entre deux essais (s)

OUT_DIR = f"data/subject_{SUBJECT}"
RAW_FILENAME = os.path.join(OUT_DIR, f"session{SESSION}_run{RUN}_raw.fif")
EVENTS_FILENAME = os.path.join(OUT_DIR, f"session{SESSION}_run{RUN}_events.csv")

# Association doigt -> touche clavier (keysym Tkinter). Le sujet appuie la
# touche AVEC le doigt nomme dans la consigne (pas besoin d'un dispositif
# externe pour dater la pression).
FINGER_KEYS = {
    "thumb_index": "h",
    "thumb_middle": "j",
    "thumb_ring": "k",
    "thumb_little": "l",
}
KEY_TO_FINGER = {v: k for k, v in FINGER_KEYS.items()}

FINGER_LABELS_FR = {
    "thumb_index": "Thumb + Index - 1",
    "thumb_middle": "Thumb + Middle - 2",
    "thumb_ring": "Thumb + Ring - 3",
    "thumb_little": "Thumb + Little - 4",
}


# ---------------------------------------------------------------------------
# Acquisition EEG continue (thread separe)
# ---------------------------------------------------------------------------
class EEGRecorder(threading.Thread):
    """Tourne en arriere-plan et accumule en continu les echantillons EEG."""

    def __init__(self, eeg: Explore_EEG):
        super().__init__(daemon=True)
        self.eeg = eeg
        self.samples = []
        self.timestamps = []
        self._stop_event = threading.Event()
        self.lock = threading.Lock()

    def run(self):
        while not self._stop_event.is_set():
            sample, timestamp = self.eeg.get_data()
            if sample is None:
                continue  # timeout ponctuel du flux, on ne casse pas l'enregistrement
            with self.lock:
                self.samples.append(sample)
                self.timestamps.append(timestamp)

    def stop(self):
        self._stop_event.set()

    def get_data_copy(self):
        with self.lock:
            return list(self.samples), list(self.timestamps)


def make_marker_outlet():
    info = StreamInfo(
        name="FingerTaskMarkers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id="finger_task_markers",
    )
    return StreamOutlet(info)


def build_trial_list():
    trials = []
    for finger in FINGER_KEYS:
        trials += [finger] * N_REPS_PER_FINGER
    if RANDOMIZE_ORDER:
        random.shuffle(trials)
    return trials


# ---------------------------------------------------------------------------
# Interface graphique de la tache (Tkinter)
# ---------------------------------------------------------------------------
class FingerTaskApp:
    def __init__(self, trials, marker_outlet, events_log):
        self.trials = trials
        self.trial_idx = 0
        self.outlet = marker_outlet
        self.events_log = events_log  # liste de dict, remplie au fil des essais

        self.expected_finger = None
        self.cue_time = None
        self.waiting_for_press = False
        self.timeout_id = None

        self.root = tk.Tk()
        self.root.title("Tache doigts - EEG")
        self.root.geometry("700x400")
        self.label = tk.Label(self.root, text="", font=("Helvetica", 36))
        self.label.pack(expand=True)
        self.info_label = tk.Label(self.root, text="", font=("Helvetica", 14), fg="gray")
        self.info_label.pack(side="bottom", pady=10)

        self.root.bind("<Key>", self.on_key)
        self.root.after(500, self.start_experiment)

    def push_marker(self, label, **extra):
        t = local_clock()
        self.outlet.push_sample([label], timestamp=t)
        row = {"lsl_time": t, "label": label}
        row.update(extra)
        self.events_log.append(row)
        return t

    def start_experiment(self):
        self.push_marker("experiment_start")
        self.run_fixation()

    def next_trial(self):
        if self.trial_idx >= len(self.trials):
            self.end_experiment()
            return
        if self.trial_idx > 0 and self.trial_idx % TRIALS_PER_BREAK == 0:
            self.show_break()
            return
        self.run_fixation()

    def show_break(self):
        self.label.config(text="Pause\n\nAppuyez sur ESPACE pour continuer")
        self.waiting_for_press = False
        self.root.bind("<space>", self.resume_after_break)

    def resume_after_break(self, event=None):
        self.root.unbind("<space>")
        self.push_marker("break_end")
        self.run_fixation()

    def run_fixation(self):
        self.label.config(text="+")
        self.info_label.config(text=f"Essai {self.trial_idx + 1}/{len(self.trials)}")
        duration = random.uniform(FIXATION_MIN, FIXATION_MAX)
        self.root.after(int(duration * 1000), self.run_cue)

    def run_cue(self):
        finger = self.trials[self.trial_idx]
        self.expected_finger = finger
        self.label.config(text=FINGER_LABELS_FR[finger])
        self.cue_time = self.push_marker(f"cue_{finger}", trial=self.trial_idx, finger=finger)
        self.waiting_for_press = True
        self.timeout_id = self.root.after(int(MAX_RESPONSE_TIME * 1000), self.on_timeout)

    def on_key(self, event):
        if not self.waiting_for_press:
            return
        keysym = event.keysym.lower()
        finger = KEY_TO_FINGER.get(keysym)
        if finger is None:
            return  # touche non pertinente pour la tache, on ignore

        self.waiting_for_press = False
        self.root.after_cancel(self.timeout_id)

        rt = local_clock() - self.cue_time
        correct = finger == self.expected_finger
        label = f"press_{finger}" if correct else f"press_{finger}_ERREUR"
        self.push_marker(
            label,
            trial=self.trial_idx,
            expected=self.expected_finger,
            pressed=finger,
            rt=rt,
            correct=correct,
        )

        self.label.config(text="" if correct else "Erreur !")
        self.trial_idx += 1
        self.root.after(int(POST_PRESS_FEEDBACK * 1000), self.run_rest)

    def on_timeout(self):
        self.waiting_for_press = False
        self.push_marker(
            f"timeout_{self.expected_finger}",
            trial=self.trial_idx,
            expected=self.expected_finger,
            pressed=None,
            rt=None,
            correct=False,
        )
        self.label.config(text="Trop lent !")
        self.trial_idx += 1
        self.root.after(int(POST_PRESS_FEEDBACK * 1000), self.run_rest)

    def run_rest(self):
        self.label.config(text="")
        self.root.after(int(INTER_TRIAL_REST * 1000), self.next_trial)

    def end_experiment(self):
        self.push_marker("experiment_end")
        self.label.config(text="Merci !\nEnregistrement termine.")
        self.info_label.config(text="")
        self.root.after(1500, self.root.destroy)

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------
def save_raw_and_events(eeg, samples, timestamps, events_log):
    data = np.array(samples).T  # (n_channels, n_times)
    info = mne.create_info(eeg.ch_names, sfreq=eeg.fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    # Les marqueurs et l'EEG partagent la meme horloge LSL : on convertit
    # simplement les temps des marqueurs en temps relatif au premier
    # echantillon EEG enregistre.
    t0 = timestamps[0]
    onsets, durations, descriptions = [], [], []
    for row in events_log:
        onset = row["lsl_time"] - t0
        if onset < 0:
            continue  # marqueur pousse avant le premier echantillon EEG (rare)
        onsets.append(onset)
        durations.append(0.0)
        descriptions.append(row["label"])

    raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
    raw.save(RAW_FILENAME, overwrite=True)
    print(f"[INFO] EEG sauvegarde : {RAW_FILENAME} ({raw.n_times / eeg.fs:.1f} s)")

    with open(EVENTS_FILENAME, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in events_log for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events_log)
    print(f"[INFO] Log des essais sauvegarde : {EVENTS_FILENAME}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[INFO] Connexion au casque Mentalab (LSL) ...")
    eeg = Explore_EEG()
    if eeg.inlet is None:
        print("[ERREUR] Impossible de se connecter au flux EEG. "
              "Verifiez que l'acquisition Mentalab est active et diffuse bien "
              "un flux LSL nomme 'Explore_8547_ExG'.")
        return

    print("[INFO] Creation du flux de marqueurs LSL 'FingerTaskMarkers' ...")
    marker_outlet = make_marker_outlet()

    print("[INFO] Demarrage de l'enregistrement EEG en continu ...")
    recorder = EEGRecorder(eeg)
    recorder.start()

    trials = build_trial_list()
    print(f"[INFO] {len(trials)} essais generes ({N_REPS_PER_FINGER} x 4 doigts, "
          f"ordre {'aleatoire' if RANDOMIZE_ORDER else 'bloque'}).")

    events_log = []
    app = FingerTaskApp(trials, marker_outlet, events_log)
    app.run()  # bloque jusqu'a la fin de la tache (fenetre fermee)

    print("[INFO] Arret de l'enregistrement EEG ...")
    recorder.stop()
    time.sleep(0.5)  # laisse le temps au thread de se terminer proprement
    samples, timestamps = recorder.get_data_copy()

    if not samples:
        print("[ERREUR] Aucune donnee EEG enregistree.")
        return

    save_raw_and_events(eeg, samples, timestamps, events_log)


if __name__ == "__main__":
    main()
