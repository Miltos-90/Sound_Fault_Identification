""" This module contains all functions that extract the harmonic features of the sound signals. """

import numpy as np
from typing import Tuple
from . import preprocessing as pre


def features(
    frequencies        : np.array, amplitudes        : np.array, 
    harmonicFrequencies: np.array, harmonicAmplitudes: np.array, 
    peakFrequencies    : np.array, peakAmplitudes    : np.array, 
    sampleFrequency: int, numHarmonics: int, axis: int) -> np.array:
    """ Extracts all harmonic features in the frequency-domain from an array of signals.
        Inputs:
            frequencies         : Frequency vector at which the amplitudes have been computed.
            amplitudes          : Array of spectral amplitudes.
            harmonicFrequencies : Array of harmonic frequencies.
            harmonicAmplitudes  : Array of harmonic amplitudes.
            peakFrequencies     : Array of peak frequencies.
            peakAmplitudes      : Array of peak amplitudes.
            sampleFrequency     : Sampling rate in Hertz.
            numHarmonics        : Number of harmonic frequencies to be extracted.
            axis                : Axis along which the amplitudes are arranged over frequencies.
        Outputs:
            features: Array of features extracted in the time-domain. The output array
                      has the same dimensions as the input signal array, with the exception of
                      axis <axis>, which contains 8 elements (i.e. features).
    """

    features = np.concatenate(
        [
            _deviation(frequencies, amplitudes, harmonicFrequencies, harmonicAmplitudes, sampleFrequency, axis),
            _inharmonicity(harmonicFrequencies, harmonicAmplitudes,  peakFrequencies, peakAmplitudes, axis),
            _noisiness(amplitudes, harmonicAmplitudes, axis),
            _oddToEvenEnergyRatio( harmonicAmplitudes, axis),
            _tristimulus(harmonicAmplitudes,   axis = axis),
        ],
        axis = axis
    )

    return features


def _energy(x: np.array, axis: int) -> np.array:
    """ Evaluates the energies of the signals contained in an axis of the input matrix in the frequency domain. 
    Inputs:
        x   : Matrix containing the spectral amplitudes along one dimension
        axis: Axis along which the computations will be performed
    Outputs: 
        Energies of the signals along the requested axis
    """
    return np.abs(x ** 2).sum(axis = axis, keepdims = True)


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
        

def _inharmonicity(
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
    peakFrequencies_ = peakFrequencies.astype(float)
    harmonicFrequencies, harmonicAmplitudes, peakFrequencies_, peakAmplitudes = \
        _truncate(axis = axis,
            x1 = harmonicFrequencies, y1 = harmonicAmplitudes, 
            x2 = peakFrequencies_,     y2 = peakAmplitudes)
    
    n        = harmonicFrequencies.shape[axis]
    harmNums = pre.expand(np.arange(n), harmonicFrequencies.ndim, axis)
    
    # If less than numHarmonics peaks have been found, the matrix is filled with zeroes. 
    # Convert those to nans so that they will be ignored by nansum() downstream
    peakFrequencies_[peakFrequencies_ == 0] = np.nan 
    fundamentalFrequencies = pre.take(harmonicFrequencies, [0], axis = axis)

    # Compute coefficient
    t   = np.abs( peakFrequencies_ - harmNums * fundamentalFrequencies ) * harmonicAmplitudes ** 2
    num = np.nansum(t, axis = axis, keepdims = True)
    den = np.nansum(harmonicAmplitudes ** 2, axis = axis, keepdims = True)
    out = num / den * 2 / fundamentalFrequencies

    return out


def _noisiness(powerAmplitudes: np.array, harmonicAmplitudes: np.array, axis: int) -> np.array:
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

    tEnergy   = _energy(powerAmplitudes, axis = axis)
    hEnergy   = _energy(harmonicAmplitudes, axis = axis)
    noisiness = (tEnergy - hEnergy) / tEnergy

    return noisiness


def _oddToEvenEnergyRatio(amplitudes: np.array, axis: int) -> np.array:
    """ Computes the odd to even harmonic energy ratio.
        Inputs: 
            amplitudes: Array of arbitrary dimensions containing the harmonic amplitudes
            axis      : Axis along which to evaluate the ratio
        Outputs:
            oer: odd-to-even harmonic energy ratio. Dimensions of the array match the dimensions
                of the input amplitudes, with the exception of axis <axis> which has been removed.
    """

    harmAmps2 = amplitudes ** 2
    numSample = amplitudes.shape[axis]
    oddHarms  = pre.take(harmAmps2, np.arange(1, numSample, 2), axis)
    evenHarms = pre.take(harmAmps2, np.arange(0, numSample - 1, 2), axis)

    num = np.sum(oddHarms, axis = axis, keepdims = True)
    den = np.sum(evenHarms, axis = axis, keepdims = True)
    oer =  num / den

    return oer


def _tristimulus(amplitudes: np.array, axis: int) -> np.array:
    """ Computes the tristimulus.
        Inputs: 
            amplitudes: Array containing amplitudes, of arbitrary dimensions.
            axis      : Axis along which the computations will be performed
        Outputs:
            out: Tristimulus coefficients. Their dimensions equal the dimensions of the
                input amplitudes, with axis <axis> containing only 3 elements (= first, 
                second, and third coefficients)
    """

    s   =  amplitudes.sum(axis)
    out = np.stack([
        pre.take(amplitudes, 1, axis) / s,                                             # t1
        pre.take(amplitudes, np.arange(2, 5), axis).sum(axis) / s,                     # t2
        pre.take(amplitudes, np.arange(5, amplitudes.shape[axis]), axis).sum(axis) / s # t3
    ], axis = axis)

    return out


def _deviation(
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
    dev  = np.abs(np.sum(harmonicAmplitudes - amps, axis = axis, keepdims = True)) / harmonicAmplitudes.shape[axis]

    return dev