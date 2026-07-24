"""
Simule un flux LSL nomme 'Explore_8547_ExG' (comme le casque Mentalab reel)
pour pouvoir tester record_finger_task.py sans materiel.

Lancez ce script dans un terminal, laissez-le tourner, puis lancez
record_finger_task.py dans un AUTRE terminal : il detectera ce flux simule
exactement comme il detecterait le vrai casque.

Arret : Ctrl+C dans ce terminal.
"""

import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 250          # frequence d'echantillonnage simulee (Hz)
N_CHANNELS = 32   # comme le casque reel


def make_info():
    info = StreamInfo(
        name="Explore_8547_ExG",
        type="EEG",
        channel_count=N_CHANNELS,
        nominal_srate=FS,
        channel_format="float32",
        source_id="simulated_explore_8547",
    )
    channels = info.desc().append_child("channels")
    for i in range(N_CHANNELS):
        ch = channels.append_child("channel")
        ch.append_child_value("label", f"ch{i + 1}")
    return info


def main():
    outlet = StreamOutlet(make_info())
    print(f"[SIM] Flux LSL simule 'Explore_8547_ExG' demarre "
          f"({N_CHANNELS} canaux, {FS} Hz). Laissez tourner ce terminal, "
          f"Ctrl+C pour arreter.")

    period = 1.0 / FS
    next_tick = time.time()
    try:
        while True:
            sample = (np.random.randn(N_CHANNELS) * 10).tolist()  # bruit aleatoire, en microvolts fictifs
            outlet.push_sample(sample)
            next_tick += period
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[SIM] Arret du flux simule.")


if __name__ == "__main__":
    main()
