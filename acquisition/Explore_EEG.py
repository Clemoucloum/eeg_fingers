
from pylsl import StreamInlet, resolve_byprop

"""
Class handling basic function with Explore EEG headset
the one used here is Explore_8547_ExG
"""


class Explore_EEG:
    def __init__(self):
        self.inlet,self.fs,self.n_channels=self.connect_EEG_stream()
        self.ch_names=self.Get_channel_names()

    def connect_EEG_stream(self):
        print("[CLIENT] Looking for Explore_8547_ExG stream ...")
        streams = resolve_byprop('name', 'Explore_8547_ExG', timeout=10)
        if len(streams) == 0:
            print("[CLIENT] Can't find Explore_8547_ExG stream.")
            return None ,None,None

        inlet = StreamInlet(streams[0])
        fs = int(inlet.info().nominal_srate())
        n_channels = inlet.info().channel_count()
        print(f"[CLIENT] Found EEG stream with {n_channels} channels at {fs}Hz")
        return inlet, fs, n_channels

    def get_data(self):
        sample, timestamp = self.inlet.pull_sample(timeout=2)
        if sample is None:
            print("[CLIENT] Stream ended (no new samples).")
            return None,None
        return sample, timestamp

    def Get_channel_names(self):
        if self.inlet is not None:
            ch_elem = self.inlet.info().desc().child('channels').first_child()
            ch_names = []
            for _ in range (self.n_channels):
                ch_names.append(ch_elem.child_value('label'))
                ch_elem = ch_elem.next_sibling()
            return ch_names
        else:
            return []

if __name__ == "__main__":
    ex = Explore_EEG()
    print(ex.connect_EEG_stream())
    print(ex.Get_channel_names())
    print(ex.get_data())