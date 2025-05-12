class Constants:
    DATA_PATH = '/Users/belial/Public/OneDrive - elte.hu/ELTE/Research_Projects/P2-Anomaly_detection/TUH_EEG_Corpus_v2.0.1/'
    SAMPLE_FREQ = 256.0
    FILTER_RANGE = (0.0, 40.0)

    @property
    def DATA_PATH(self):
        return self.DATA_PATH

    @property
    def SAMPLE_FREQ(self):
        return self.SAMPLE_FREQ

    @property
    def FILTER_RANGE(self):
        return self.FILTER_RANGE[0], self.FILTER_RANGE[1]

constants = Constants()