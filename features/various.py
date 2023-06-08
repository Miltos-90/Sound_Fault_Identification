""" This module contains all functions that extract the features of a sound signal that do not
    belong to the other categories.
"""

import numpy as np

def tonality(amplitudes: np.array, axis: int) -> np.array:
    """ Computes the tonality coefficient of a spectrum.
        Inputs:
            amplitudes: Array of power amplitudes
            axis      : Axis along which to compute the coefficient
        Outputs:
            tonality: Tonality coefficient(s). The dimensions match the dimensions
                      of the input amplitudes array, with axis <axis> missing.
    """

    eps            = np.finfo(float).eps
    geometricMean  = np.exp(np.mean(np.log(amplitudes + eps), axis = axis, keepdims = True))
    arithmeticMean = np.mean(amplitudes, axis = axis, keepdims = True)

    # Spectral flatness
    flatness = geometricMean / (arithmeticMean + eps)
    flatness = 10 * np.log10(flatness) # Convert to dB

    # Evaluate tonality coefficient
    tonal = np.minimum(flatness / -60.0, 1)

    return tonal


