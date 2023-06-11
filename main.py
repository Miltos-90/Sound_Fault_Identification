
import features
import contextlib
import utils
import numpy as np
import multiprocessing
import joblib
import os
from tqdm import tqdm

SAMPLE_FREQUENCY = 16000 # Sample frequency [Hz]


@contextlib.contextmanager
def tqdmJoblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
        https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697 
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def featureExtractor(filepath, fs = SAMPLE_FREQUENCY, axis = 0):

    # Make path to output features file (simply change the extension)
    outPath = filepath.strip('.wav') + '.npy'

    # If the output file does not exist, load the signal, process it
    # and save the extracted features
    if not os.path.isfile(outPath):
        signal = utils.readWav(filepath, signalType = 'float')
        feats  = features.extract(signal, fs, axis = axis)
        np.save(outPath, feats)

    return
    

if __name__ == '__main__':

    numCores = 6
    folder   = './data'
    
    df = utils.indexFiles(folder)
    numSamples = df.shape[0]
    
    with tqdmJoblib( tqdm(desc = "Extracting features", total = numSamples)) as progressBar:
        joblib.Parallel(n_jobs = numCores)(
            joblib.delayed(featureExtractor)(fPath) for fPath in df['filepath'].values)

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