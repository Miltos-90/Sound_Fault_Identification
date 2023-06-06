""" This module contains all functions that extract the harmonic features of the sound signals. """

import numpy as np
from typing import Tuple
from . import preprocessing as pre

def energy(x: np.array, axis: int) -> np.array:
    """ Evaluates the energies of the signals contained in an axis of the input matrix in the frequency domain. 
    Inputs:
        x   : Matrix containing the spectral amplitudes along one dimension
        axis: Axis along which the computations will be performed
    Outputs: 
        Energies of the signals along the requested axis
    """
    return np.abs(x ** 2).sum(axis = axis)


def _truncate(x1: np.array, y1: np.array, x2: np.array, y2: np.array, axis: int
    ) -> Tuple[np.array, np.array, np.array, np.array]:
    """ Aligns the dimensions of two matrix (x, y) pairs by truncating the longest pair.
        Inputs:
            x1, y1: First pair of matrices
            x2, y2: Second pair of matrices
            axis: Axis along which the pairs will be dimension-aligned
        Outputs
            x1_, y1_, x2_, y2_: Dimension-aligned patrix pairs. Their dimensions equal the
            dimensions of the input matrices, with the exception of axis <axis>. This will
            have a number of elements that equals the lowest number of elements 
    """

    n1, n2 = x1.shape[axis], x2.shape[axis]
    cut    = lambda arr, num: pre.take(arr, indices = np.arange(num), axis = axis)

    if n2 < n1: # Pair 2 has fewer elements than pair 1 along axis <axis>
        x1_, y1_ = cut(x1, n2), cut(y1, n2)
        x2_, y2_ = x2, y2
    elif n1 < n2: # Pair 1 has fewer elements than pair 2
        x2_, y2_ = cut(x2, n1), cut(y2, n1)
        x1_, y1_ = x1, y1
    else: # Already aligned. Nothing to do
        x1_, y1_, x2_, y2_ = x1, y1, x2, y2
    
    return x1_, y1_, x2_, y2_
        

def inharmonicity(
    harmonicFrequencies: np.array, harmonicAmplitudes: np.array, peakFrequencies: np.array, 
    peakAmplitudes: np.array, axis: int) -> np.array:
    """ Computes a signal's inharmonicity, i.e. the divergence of the signal's spectral components
        from a purely harmonic signal.
        Inputs:
            harmonicFrequencies    : Array of harmonic frequencies
            harmonicAmplitudes     : Array of harmonic amplitudes
            peakFrequencies        : Array of peak amplitudes
            peakAmplitudes         : Array of peak amplitudes
            axis                   : Axis along which to perform the computations
            NOTE: All matrices can have an arbitrary number of dimensions, as long as they are the same
                  for all.
        Outputs:
            Array of inharmonicity coefficients. The dimensions match the input dimensions, with axis
            <axis> having been removed.

    """
            
    harmonicFrequencies, harmonicAmplitudes, peakFrequencies, peakAmplitudes = \
        _truncate(axis = axis,
            x1 = harmonicFrequencies, y1 = harmonicAmplitudes, 
            x2 = peakFrequencies,     y2 = peakAmplitudes)
    
    n        = harmonicFrequencies.shape[axis]
    harmNums = pre.expand(np.arange(n), harmonicFrequencies.ndim, axis)
    
    # Compute coefficient
    # If less than numHarmonics peaks have been found, the matrix is filled with zeroes. 
    # Convert those to nans so that they will be ignored by nansum()
    peakFrequencies[peakFrequencies == 0] = np.nan 
    fundamentalFrequencies = pre.take(harmonicFrequencies, [0], axis = axis)

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
    oddHarms  = pre.take(harmAmps2, np.arange(1, numSample, 2), axis)
    evenHarms = pre.take(harmAmps2, np.arange(0, numSample - 1, 2), axis)
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
    t1 = pre.take(amplitudes, 1, axis) / s
    t2 = pre.take(amplitudes, np.arange(2, 5), axis).sum(axis) / s
    t3 = pre.take(amplitudes, np.arange(5, amplitudes.shape[axis]), axis).sum(axis) / s

    return t1, t2, t3


def deviation(
    frequencies: np.array, amplitudes: np.array, harmonicFrequencies: np.array, 
    harmonicAmplitudes: np.array, sampleFrequency: int, axis: int):
    """ Computes the harmonic spectral deviation (i.e. the deviation of the amplitude harmonic peaks
        from the global spectral envelope).
        Inputs:
            amplitudes           : Matrix containing the spectral amplitudes
            frequencies          : Vector containing the frequencies corresponding to the spectral amplitudes
            harmonicFrequencies  : Array of harmonic frequencies
            harmonicAmplitudes   : Array of harmonic amplitudes
            sampleFrequency      : Sampling rate in Hertz
            axis                 : Axis along which to compute the spectral descriptors
        Outputs:
            dev : Harmonic spectral deviation of the amplitudes along axis <axis>
    """

    f0   = pre.take(harmonicFrequencies, [0], axis = axis) # Extract fundamental frequency
    env  = pre.envelopes.spectral(amplitudes, sampleFrequency, fundamentalFrequency = f0, axis = axis)
    amps = pre.extract(frequencies, env, harmonicFrequencies, axis = axis)
    dev  = np.abs(np.sum(harmonicAmplitudes - amps, axis = axis)) / harmonicAmplitudes.shape[axis]

    return dev