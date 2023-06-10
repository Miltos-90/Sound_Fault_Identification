
"""
Profiler call:
python -m cProfile -s tottime ./main.py > profiler.txt
"""


import features
import utils
import numpy as np
df = utils.indexFiles('./data/-6_dB_fan')

for _, dfChunk in df.groupby(np.arange(len(df)) // 10):
    
    signals = []
    for f in dfChunk['filepath'].values:
        sampleFrequency, signal = utils.readWav(f, signalType = 'float')
        signals.append(signal)

    break


signals = np.stack(signals, axis = -1)
signals = signals[0:80000, 0:2, 0:2]
f = features.extract(signals, sampleFrequency, axis = 0)
print(f'Signal shape {signals.shape}. Feature shape {f.shape}. Nans {np.isnan(f).sum() / f.size * 100}')


# CHECK IF IT WORKS IN ALL DIMENSION AND SIGNAL ARRAYS
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
