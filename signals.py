""" This module generates sample signals in time-domain.
"""

import numpy as np
from scipy import signal

def signal1(
    N: int,      # Number of samples in the time domain 
    f0: int,     # Fundamental frequency [Hz]
    fs: int,     # Sampling rate [Hz]
    noise: float # RMS Noise amplitude [-]
    ) -> np.array:

    n    = np.arange(N);                    # time in samples
    fund = np.sin(2 * np.pi * f0 * n / fs)  # sine at fundamental frequency
    sig  = np.zeros(N)

    # Make harmonics
    npartials = 7
    for k in range(1, npartials + 1):
        ampk = 1 / k ** 2  # give a natural roll-off
        sig += ampk * np.sin(2 * np.pi * k * f0 * n / fs) 

    # Notch out fundamental frequency
    b, a = [-2 * np.cos(2 * np.pi * f0 / fs), 1], [1]
    sig  = signal.lfilter(b, a, sig)

    # Normalize
    sig /= sig.max()

    # Add some noise for realism
    sig     += noise * np.random.randn(N) 

    return sig


def signal2(
    N: int,      # Number of samples in the time domain 
    f0: int,     # Fundamental frequency [Hz]
    fs: int,     # Sampling rate [Hz]
    noise: float # RMS Noise amplitude [-]
    ) -> np.array:

    F      = np.array([700, 1220, 2600])  # Formant frequencies in Hz
    B      = np.array([130, 70, 160])     # Formant bandwidths in Hz
    R      = np.exp(-np.pi * B / fs)      # Pole radii
    theta  = 2 * np.pi * F / fs           # Pole angles
    poles  = R * np.exp(1j * theta)
    b, a   = signal.zpk2tf([], np.concatenate((poles, np.conj(poles))), 1)
    w0T    = 2 * np.pi * f0 / fs
    nharm  = int(np.floor((fs / 2) / f0))  # number of harmonics
    sig    = np.zeros(N)
    n      = np.arange(N)

    # Synthesize bandlimited impulse train:
    for i in range(1, nharm + 1):
        sig += np.cos(i * w0T * n)  

    # Low=pass and normalize
    sig = signal.lfilter(b, a, sig)
    sig /= np.max(sig)

    # Add noise for realism
    sig += noise * np.random.randn(N)

    return sig