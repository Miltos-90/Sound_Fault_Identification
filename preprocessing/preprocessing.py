"""
This module implements the preprocessing steps mentioned in Section 2 of:

"A large set of audio features for sound description (similarity and classification)
in the CUIDADO project", Peeters G., 2004.

URL: http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
(accessed 11/05/2023)
"""

from scipy.signal.windows import hann
from scipy.signal import hilbert, firwin2, freqz
from scipy.fft import fft, fftfreq
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


# =================== Envelope analysis related functions =====================

# =========================== Additional functions =============================
def sinusoidalHarmonicModel(frequencies: np.array, amplitudes: np.array, harmonics: int, axis: int):
    """ Extracts (complex) amplitudes and frequencies of the sinusoids that best approximate a signal, by 
        analyzing the signal's spectrum and identifying the frequencies and amplitued of its spectral peaks.
        Inputs: 
            frequencies: Vector of frequencies for the corresponding FFT amplitudes
            amplitudes : FFT amplitudes [Num. frames x Num. frequencies x Num. channels]
            harmonics  : Number of harmonic frequencies and amplitudes to extract
            axis       : Axis along which to search for the harmonic amplitudes and frequencies
        Outputs:
            harmonicFreqs: Harmonic frequencies [Num. frames x harmonics x Num. channels]
            harmonicAmps:  Harmonic amplitudes  [Num. frames x harmonics x Num. channels]
    """

    # Compute the fundamental frequency
    peakIndex   = np.argmax(np.abs(amplitudes), axis = axis)
    fundamental = frequencies[peakIndex]

    # Compute sinusoidal harmonic frequencies and corresponding amplitudes
    frames        = amplitudes.shape[0]
    channels      = amplitudes.shape[2]
    shape         = (frames, harmonics, channels)
    harmonicFreqs = np.empty(shape = shape, dtype = frequencies.dtype)
    harmonicAmps  = np.empty(shape = shape, dtype = amplitudes.dtype)
    freqExpanded  = np.expand_dims(frequencies, axis = tuple(range(1, amplitudes.ndim)))

    for hNo in range(1, harmonics + 1):

        # Get the spectrum frequency that is closest to the current harmonic frequency
        diff = freqExpanded - fundamental * hNo
        idx  = np.abs(diff).argmin(axis = 0)
        harmonicFreqs[:, hNo-1, :] = frequencies[idx]

        # get the corresponding amplitude
        idx     = np.expand_dims(idx, axis = axis)
        curAmps = np.take_along_axis(amplitudes, idx, axis = axis)
        harmonicAmps[:, hNo-1, :] = np.squeeze(curAmps, axis = axis)
        
    return harmonicFreqs, harmonicAmps
