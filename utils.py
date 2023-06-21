""" Set of utilities-related functions. """

import os
import numpy as np 
import pandas as pd
from scipy.io import wavfile
from features import extract


def featureExtractor(filepath: str, fs: int = 16000, axis: int = 0):
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
        feats  = extract(signal, fs, axis = axis)
        np.save(outPath, feats)

    return


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


