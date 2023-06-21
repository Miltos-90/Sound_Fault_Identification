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
    f0            = np.take(hFreqs, [0], axis = axis) # 1st harmonic frequency = fundamental frequency

    # If less than the requested number of peaks is found, matrices of a smaller size
    # will be returned from the peakFinder() function. This might be problematic for bulk
    # extraction of data (i.e. from multiple signals). This is why new arrays will be 
    # created for the peak amplitudes and frequencies that will contain the expected
    # number (= numHarmonics) of elements, filled with nans.
    pFreqs = _padArray(pFreqs, numHarmonics, axis)
    pAmps  = _padArray(pAmps,  numHarmonics, axis)

    return np.concatenate([maxAmplitude, pFreqs, pAmps, f0], axis = axis)


def _padArray(
    arr: np.array, numElements: int, axis: int, value: float = np.nan, direction: str = 'end'
    ) -> np.array:
    """ Pads an array along an axis over one direction with a constant value to achieve a specific
        number of elements over that dimension.
        Inputs:
            arr  : Array to be padded
            numElements: Required number of elements that the array should have after padding
            axis : Axis along which the padding will take place
            value: Value with which additional elements will be padded
        
        Outputs:
            arr_: padded array
    """
    
    # Evaluate the padwidth that is required to align the dimensions

    padWidths = []
    for dim in range(arr.ndim):
        
        if dim == axis:
            add = numElements - arr.shape[axis]
            if add < 0: raise ValueError('Negative pad width encountered.')
            if   direction == 'start': padWidth = (add, 0) # insert elements to the start
            elif direction == 'end'  : padWidth = (0, add) # append elements to the end
            else: raise ValueError('Direction can be set to either "start" or "end".')
        else:
            padWidth = (0, 0)
        
        padWidths.append(padWidth)

    # Make new array
    arr_ = np.pad(arr, padWidths, 'constant', constant_values = value)
    
    return arr_