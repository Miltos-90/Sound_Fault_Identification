""" Set of utilities-related functions. """

import os
import numpy as np 
import pandas as pd
from scipy.io import wavfile


def makeIndex(filePath: str):
    """ Indexes all files and extracts some metadata and the target. """

    data = list()

    for root, dirs, files in os.walk(filePath):

        for filename in files:
            nm, ext = os.path.splitext(filename)

            if ext.lower().endswith('.wav'):

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

    fs, data = wavfile.read(filepath) # Read file
    if micID is not None: 
        data = data[:, micID]   # Return specific mic. record
        data = data[:, None]    # Add a new axis for shape consistency

    if signalType: data = data.astype(signalType)
    return fs, data


