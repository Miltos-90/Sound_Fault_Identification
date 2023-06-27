
""" Constants used in utils.py """
FS = 16000               # [Hz] Sampling rate
MISSING_THRESHOLD = 0.05 # [%] Missing value threshold above which features are removed
FILL_VALUE = 0.0         #     Value to replace the missing values in the dataset
STD_THRESHOLD = 0.05     # [%] Threshold below which a feature is considered constant
CORR_THRESHOLD = 0.95    # [-] Threshold above which highly correlated features (Spearman rank correlation) are removed

""" Constants used in utils.py -> featureExtractor() """
FRAME_SIZE = 2 ** 12 # Size of the frames that the signal is split into      
hopSize = 2 ** 10 # Hop size of the adjacent frames the signal is split into
envFrameSize = 2 ** 9 # Size of the frames that the signal's RMS envelope is split into
envHopSize = 2 ** 6 # Hop of the frames that the signal's RMS envelope is split into
padType = 'center' # Type of padding to be applied in order to split the signal in frames of the input frame- and hop- size.
padValue = 0.0 # Value to pad the signals with if needed
numFilters = 24 # Number of Bark bands to be used for the Bark spectrogram
harmonics = 10 # Number of harmonic frequencies to extract
numMFCC = 12 # Number of Mel Frequency Cepstral Coefficients to extract
numLags = 12 # Number of autocorrelation lags to extract
octaveDesignator = 3