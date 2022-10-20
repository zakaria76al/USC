class params:
    # Data parameters
    sample_rate = 16000
    nb_classes = 12
    frame_size = 15360  # seconds * sample_rate
    hop_size = 7680  # seconds * sample_rate
    sequence_nbr = 6
    mfcc_coefficients = 40

    # Model parameters
    tower_nbr = 40
    filter1 = 128
    filter2 = 64
    filter3 = 32

    def __init__(self, sample_rate=sample_rate, nb_classes=nb_classes, frame_size=frame_size, hop_size=hop_size,
                 sequence_nbr=sequence_nbr, mfcc_coefficients=mfcc_coefficients, tower_nbr=tower_nbr, filter1=filter1,
                 filter2=filter2, filter3=filter3):
        self.sample_rate = sample_rate
        self.nb_classes = nb_classes
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sequence_nbr = sequence_nbr
        self.mfcc_coefficients = mfcc_coefficients
        self.tower_nbr = tower_nbr
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3
