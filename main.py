
import features
import contextlib
import utils
import numpy as np
import multiprocessing
import joblib
import os
from tqdm import tqdm
import pandas as pd

SAMPLE_FREQUENCY = 16000 # Sample frequency [Hz]




def featureExtractor(filepath, fs = SAMPLE_FREQUENCY, axis = 0):

    # Make path to output features file (simply change the extension)
    outPath = filepath.strip('.wav') + '.npy'

    # If the output file does not exist, load the signal, process it
    # and save the extracted features
    #if not os.path.isfile(outPath):
    try:
        signal = utils.readWav(filepath, signalType = 'float')
        feats  = features.extract(signal, fs, axis = axis)
        np.save(outPath, feats)
    except:
        print(filepath)

    return
    

if __name__ == '__main__':

    numCores = 6
    folder   = './data'
    
    #df = utils.indexFiles(folder)
    df = pd.read_csv('./out.csv', index_col = 0)
    numSamples = df.shape[0]
    
    with features.tqdmJoblib( tqdm(desc = "Extracting features", total = numSamples)) as progressBar:
        joblib.Parallel(n_jobs = numCores)(
            joblib.delayed(featureExtractor)(fPath.strip('.npy') + '.wav') for fPath in df['filepath'].values)

    #f = 'C://Users//kalika01//Desktop//MIMII - Sound dataset for malfunctioning industrial//data//6_dB_slider//slider//id_00//abnormal//00000057.wav'
    #signal = utils.readWav(f, signalType = 'float')
    #feats  = features.extract(signal, fs = SAMPLE_FREQUENCY, axis = 0)
    #print(feats[:,0])

"""
# CHECK IF IT WORKS IN ALL DIMENSION AND SIGNAL ARRAYS

df = utils.indexFiles('./data/-6_dB_fan')

for _, dfChunk in df.groupby(np.arange(len(df)) // 10):
    
    signals = []
    for f in dfChunk['filepath'].values:
        sampleFrequency, signal = utils.readWav(f, signalType = 'float')
        signals.append(signal)

    break

signals = np.stack(signals, axis = -1)
signals = signals[0:80000, 0:2, 0:2]
f = features.extract(signals, SAMPLE_FREQUENCY, axis = 0)
print(f'Signal shape {signals.shape}. Feature shape {f.shape}. Nans {np.isnan(f).sum() / f.size * 100}')

signals = signals.swapaxes(0, 1)
f = features.extract(signals, sampleFrequency, axis = 1)
print(f'Signal shape {signals.shape}. Feature shape {f.shape}. Nans {np.isnan(f).sum() / f.size * 100}')

signals = signals.swapaxes(1, 2)
f = features.extract(signals, sampleFrequency, axis = 2)
print(f'Signal shape {signals.shape}. Feature shape {f.shape}. Nans {np.isnan(f).sum() / f.size * 100}')

signals = signals[0,:,:]
f = features.extract(signals, sampleFrequency, axis = 1)
print(f'Signal shape {signals.shape}. Feature shape {f.shape}. Nans {np.isnan(f).sum() / f.size * 100}')

signals = signals.swapaxes(0, 1)
f = features.extract(signals, sampleFrequency, axis = 0)
print(f'Signal shape {signals.shape}. Feature shape {f.shape}. Nans {np.isnan(f).sum() / f.size * 100}')

signals = signals[:, 0]
f = features.extract(signals, sampleFrequency, axis = 0)
print(f'Signal shape {signals.shape}. Feature shape {f.shape}. Nans {np.isnan(f).sum() / f.size * 100}')
"""