""" This module contains all functions that extract the harmonic features of a sound signal. """

import numpy as np
from .preprocessing.helpers import take
from typing import Tuple


def energy(x: np.array, axis: int) -> np.array:
    """ Evaluates the energies of the signals contained in an axis of the input matrix in the frequency domain. 
    Inputs:
        x   : Matrix containing the spectral amplitudes along one dimension
        axis: Axis along which the computations will be performed
    Outputs: 
        Energies of the signals along the requested axis
    """
    return np.abs(x ** 2).sum(axis = axis)


def inharmonicity(
    harmonicFrequencies: np.array, harmonicAmplitudes: np.array, peakFrequencies: np.array, 
    peakAmplitudes: np.array, fundamentalFrequencies: np.array, axis: int) -> np.array:
    """ Computes a signal's inharmonicity, i.e. the divergence of the signal's spectral components
        from a purely harmonic signal.
        Inputs:
            harmonicFrequencies    : Array of harmonic frequencies
            harmonicAmplitudes     : Array of harmonic amplitudes
            peakFrequencies        : Array of peak amplitudes
            peakAmplitudes         : Array of peak amplitudes
            fundamentalFrequencies : Array of fundamental amplitudes
            axis                   : Axis along which to perform the computations
            NOTE: All matrices can have an arbitrary number of dimensions, as long as they are the same
                  for all.
        Outputs:
            Array of inharmonicity coefficients. The dimensions match the input dimensions, with axis
            <axis> having been removed.

    """

   # Expand/align dimensions
    exAx     = [a for a in list(range(harmonicFrequencies.ndim)) if a != axis]
    harmNums = np.expand_dims(np.arange(harmonicFrequencies.shape[axis]), axis = exAx)
    
    # Compute coefficient
    # If less than numHarmonics peaks have been found, the matrix is filled with zeroes. 
    # Convert those to nans so that they will be ignored by nansum()
    peakFrequencies[peakFrequencies == 0] = np.nan 
    t   = np.abs( peakFrequencies - harmNums * fundamentalFrequencies ) * harmonicAmplitudes ** 2
    num = np.nansum(t, axis = axis)
    den = np.nansum(harmonicAmplitudes ** 2, axis = axis)
    out = num / den * 2 / fundamentalFrequencies

    return out


def noisiness(powerAmplitudes: np.array, harmonicAmplitudes: np.array, axis: int) -> np.array:
    """ Computes the noisiness of the signals from their harmonic and power amplitudes.
        Inputs:
            powerAmplitudes   : Array containing the power amplitudes of the signals
            harmonicAmplitudes: Array containing the harmonic amplitudes of the signals
            axis              : Axis along which to compute the noisiness
            NOTE: The signals can have an arbitrary number of dimensions, but the elements along 
                  all dimensions should be the same.
        Outputs:
            Noisiness of the signals along axis <axis>. Output dimensions match the input dimensions,
            with the exception of axis <axis> which is removed.
    """

    tEnergy   = energy(powerAmplitudes, axis = axis)
    hEnergy   = energy(harmonicAmplitudes, axis = axis)
    noisiness = (tEnergy - hEnergy) / tEnergy

    return noisiness


def oddToEvenEnergyRatio(amplitudes: np.array, axis: int) -> np.array:
    """ Computes the odd to even harmonic energy ratio.
        Inputs: 
            amplitudes: Array of arbitrary dimensions containing the harmonic amplitudes
            axis: Axis along which to evaluate the ratio
        Outputs:
            oer: odd-to-even harmonic energy ratio. Dimensions of the array match the dimensions
                of the input amplitudes, with the exception of axis <axis> which has been removed.
    """

    harmAmps2 = amplitudes ** 2
    numSample = amplitudes.shape[axis]
    oddHarms  = take(harmAmps2, np.arange(1, numSample, 2), axis)
    evenHarms = take(harmAmps2, np.arange(0, numSample - 1, 2), axis)
    oer       = np.sum(oddHarms, axis = axis) / np.sum(evenHarms, axis = axis)

    return oer


def tristimulus(amplitudes: np.array, axis: int) -> Tuple[np.array, np.array, np.array]:
    """ Computes the tristimulus.
        Inputs: 
            amplitudes: Array containing amplitudes, of arbitrary dimensions.
            axis: Axis along which the computations will be performed
        Outputs:
            t1, t2, t3: Tristimulus coefficients. Their dimensions equal the dimensions of the
            input amplitudes, with axis <axis> removed.
    """

    s  =  amplitudes.sum(axis)
    t1 = take(amplitudes, 1, axis) / s
    t2 = take(amplitudes, np.arange(2, 5), axis).sum(axis) / s
    t3 = take(amplitudes, np.arange(5, amplitudes.shape[axis]), axis).sum(axis) / s

    return t1, t2, t3

