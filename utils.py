""" Set of utilities-related functions. """

import os
import numpy as np 
import pandas as pd
from scipy.io import wavfile
from features import extract
import config


def indexFiles(filePath: str, extension: str = '.wav'):
    """ Indexes all files and extracts some metadata and the target. """

    data = list()

    for root, _, files in os.walk(filePath):

        for filename in files:
            _, ext = os.path.splitext(filename)

            if ext.lower().endswith(extension):

                fullpath = os.path.join(os.path.abspath(root), filename)
                rootPart = root.split('\\')
                target   = rootPart[-1]                   # Normal / abnormal
                machType = rootPart[-3]                   # Machine type
                machId   = int(rootPart[-2].strip('id_')) # Machine ID
                noise    = rootPart[-4].strip(            # Background noise level
                    '_'.join(machType))

                data.append((fullpath, machId, machType, noise, target))

    cols = ['filepath', 'machine_id', 'machine_type', 'background_noise', 'target']

    return pd.DataFrame(data, columns = cols)


def readWav(filepath: str, micID: int = None, signalType = None):
    """ Reads a wav file recorded by microphone <micID>. If <micID> is 
        not specified, it returns the recordings of all microphones.
    """

    _, data = wavfile.read(filepath) # Read file
    if micID is not None: 
        data = data[:, micID]   # Return specific mic. record
        data = data[:, None]    # Add a new axis for shape consistency

    if signalType: data = data.astype(signalType)
    return data


def featureExtractor(filepath: str, axis: int = 0):
    """ Wraps the feature extractor (see feeatures folder) to read an input .wav file,
        extract the features, and save thm on an .npy file.
        Inputs:
            filepath: Path to .wav file
            fs      : Sampling rate in Hertz
            axis    : Axis along which to extract the features from the signal
                      contained in the .wav file
        Outputs: None
    """

    # Make path to output features file (simply change the extension)
    outPath = filepath.strip('.wav') + '.npy'

    # If the output file does not exist, load the signal, process it
    # and save the extracted features
    if not os.path.isfile(outPath):
        signal = readWav(filepath, signalType = 'float')
        feats  = extract(
            signal = signal, 
            fs = config.FS, 
            frameSize = config.FRAME_SIZE, 
            hopSize = 2 ** 10, 
            envFrameSize = 2 ** 9,  
            envHopSize = 2 ** 6, 
            padType = 'center', 
            padValue = 0.0, 
            numFilters = 24, 
            harmonics = 10, 
            numMFCC = 12, numLags = 12, 
            octaveDesignator = 3,
            axis = axis)
        np.save(outPath, feats)

    return


def handleMissingValues(
    X:np.array, threshold: float = config.MISSING_THRESHOLD, fillValue: float = config.FILL_VALUE) -> np.array:
    """ Handles the missing values of a predictor matrix X. It will drop features (columns) with 
        a percentage of missing values higher than a threshold, and will fill the remaining the 
        remaining missing values with a constant. 
        
        Inputs:
            X        : Predictor matrix (dimensions: Num. samples x Num. features)
            threshold: Thresholf of missing values (per feature) above which the feature will
                       be dropped.
            fillValue: Value to replace the missing data from the remaining columns of the matrix.
        Outputs:
            X        : Predictor matrix without missing values (dimensions: Num. samples x Num. features).
    """

    missing = X.loc[:, X.isnull().any()] # Isolate columns with missing values only
    missingPercentage = missing.isnull().sum(axis = 0) / X.shape[0] * 100

    # These features (names) will be dropped
    dropFeats = missingPercentage.index[missingPercentage > threshold]
    missing   = missing.drop(dropFeats, axis = 1)

    X.drop(dropFeats, axis = 1, inplace = True)
    X.fillna(value = fillValue, inplace = True)

    return X


def removeApproxConstant(X: np.array, threshold: float = config.STD_THRESHOLD) -> np.array:
    """ Removes features whose variance over the dataset lies below a threshold.
        Inputs:
            X        : Array containing the predictors (dimensions: Num. samples x Num. features).
            threshold: Lower threshold of the standard deviation below which a feature
                       is removed [% of the highest standard deviation on the min-max normalized X]
        Outputs:
            X: Predictor array with low-variability features removed
    """

    XNorm = ( X - X.min() ) / ( X.max() - X.min() )
    sd    = XNorm.std()
    sd.fillna(0.0, inplace = True) # For constant features, the (max() - min()) division will result in nans
    sd = sd.sort_values(ascending = False)

    dropFeats = sd[sd < sd.max() * threshold].index
    X.drop(dropFeats, axis = 1, inplace = True)

    return X


def removeCorrelated(X: np.array, threshold: float = config.CORR_THRESHOLD) -> np.array:
    """ Removes highly correlated features, based on the Spearman rank
        coefficient.
        Inputs:
            X        : Predictor array (dimensions: num. Samples x Num. features)
            threshold: Threshold of feature pairs above which one feature of
                       the pair is removed.
        Outputs:
            X: Predictor array with highly correlated features removed
    """

    corr = X.corr(method = 'spearman')
    corr.to_csv('./spearman_correlations.csv')

    # Convert to long-format matrix, i.e. column 1, column 2, spearman coefficient
    corrPairs = (corr.abs().where(np.triu(np.ones(corr.shape), k = 1)
                    .astype(bool))
                    .stack()
                    .sort_values(ascending = False))

    # List all pairs with Spearman correlation above the threshold
    corrPairs = corrPairs[corrPairs > threshold].reset_index()

    # Extract column pairs
    col1      = set(corrPairs['level_0'].values)
    col2      = set([int(c) for c in corrPairs['level_1'].values])

    # Get columns to be removed
    dropFeats = list(col2.difference(col1))

    # Drop them
    X.drop(dropFeats, axis = 1, inplace = True)

    return X