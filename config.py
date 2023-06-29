""" Configuration file. Declares all constants used throughout.
"""

import numpy as np

""" Constants used in utils.py """
FS                = 16000       # [Hz] Sampling rate
MISSING_THRESHOLD = 0.05        # [%] Missing value threshold above which features are removed
FILL_VALUE        = 0.0         # [-] Value to replace the missing values in the dataset
STD_THRESHOLD     = 0.05        # [%] Threshold below which a feature is considered constant
CORR_THRESHOLD    = 0.95        # [-] Threshold above which highly correlated features (Spearman rank correlation) are removed

""" Constants used in utils.py -> featureExtractor() """
FRAME_SIZE          = 2 ** 12   # [-] Size of the frames that the signal is split into      
HOP_SIZE            = 2 ** 10   # [-] Hop size of the adjacent frames the signal is split into
ENVELOPE_FRAME_SIZE = 2 ** 9    # [-] Size of the frames that the signal's RMS envelope is split into
ENVELOPE_HOP_SIZE   = 2 ** 6    # [-] Hop of the frames that the signal's RMS envelope is split into
PAD_TYPE            = 'center'  # [-] Type of padding to be applied in order to split the signal in frames of the input frame- and hop- size.
PAD_VALUE           = 0.0       # [-] Value to pad the signals with if needed
NUM_FILTERS         = 24        # [-] Number of Bark bands to be used for the Bark spectrogram
HARMONICS           = 10        # [-] Number of harmonic frequencies to extract
NUM_MFCC            = 12        # [-] Number of Mel Frequency Cepstral Coefficients to extract
NUM_LAGS            = 12        # [-] Number of autocorrelation lags to extract
OCTAVE              = 3         # [-] Octave band designator for full spectra extraction

""" Constants used in engine.py """

TEST_SIZE = 0.25  # [%] Test sie sample - percentage of entire dataset
NUM_FOLDS = 5     # Number of folds for cross-validation
NUM_RS    = 500   # Number of iterations for (random) hyperparameter search
SEED      = 321   # Seed for data split and lightgbm

# Hyperparameter space for random search
HYPER_SPACE = {
    'num_leaves'      : np.logspace(3, 8, 9, base = 2.0).astype(int),     # Default: 31
    'learning_rate'   : np.logspace(-3, -1/3, 10),                        # Default: 0.1
    'max_depth'       : np.append(-1, np.linspace(3, 16, 7).astype(int)), # Default: -1
    'feature_fraction': np.linspace(0.2, 1.0, 9).round(2),                # Default: 1.0 
    'bagging_fraction': np.linspace(0.2, 1.0, 9).round(2),                # Default: 1.0
    'num_boost_round' : np.logspace(3, 8, 9, base = 2.0).astype(int),     # Default: 100
    }

# Fixed parameters for lightGBM's training
FIXED_PARAMS = {
    'objective'    : 'multiclass',
    'metric'       : 'multi_logloss',
    'is_unbalance' : True,
    'num_classes'  : 5,
    'num_threads'  : 10,
    'boosting'     :'dart',
    'verbosity'    : -1,
    'seed'         : SEED,
    'early_stopping_rounds': 5,
    }