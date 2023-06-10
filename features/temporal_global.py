""" This module contains all functions that extract the global spectral & harmonic features of the sound signals. """

import numpy as np
from . import preprocessing as pre

def features(frequencies: np.array, amplitudes: np.array, numHarmonics: int, axis: int) -> np.array:
    """ Extracts all spectral features in the frequency-domain from an array of signals.
        Inputs:
            frequencies : Frequency vector at which the amplitudes have been computed.
            amplitudes  : Array of spectral amplitudes.
            numHarmonics: Number of harmonic frequencies to be extracted.
            axis        : Axis along which the features should be extracted
        Outputs:
            features: Array of features extracted in the time-domain. The output array
                      has the same dimensions as the input signal array, with the exception of
                      axis <axis>, which contains (2 * numHarmonics + 2) elements (i.e. features).
    """

    maxAmplitude  = amplitudes.max(axis = axis, keepdims = True)
    amplitudesDB  = pre.powerTodb(amplitudes, maxAmplitude) 
    pFreqs, pAmps = pre.peakFinder(frequencies, amplitudesDB, numHarmonics, axis)
    hFreqs, _     = pre.harmonicModel(frequencies, amplitudesDB, pFreqs, numHarmonics, axis)
    f0            = pre.take(hFreqs, [0], axis = axis) # 1st harmonic frequency = fundamental frequency

    return np.concatenate([maxAmplitude, pFreqs, pAmps, f0], axis = axis)