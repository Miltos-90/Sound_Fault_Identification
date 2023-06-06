""" This module contains all functions that extract the spectral features of the sound signals. """

from . import preprocessing as pre
from scipy.fftpack import dct
from typing import Tuple
import numpy as np


def _normalizeSpectrum(frequencies: np.array, amplitudes: np.array, axis: int) -> np.array:
    """ Normalizes the amplitudes of a spectrum. 
        Inputs:
            frequencies : Matrix containing the frequencies corresponding to the spectral amplitudes
            amplitudes  : Matrix containing the spectral amplitudes
            axis        : Axis along which to normalize
            NOTE: frequencies and amplitudes are assumed to have the same number of dimensions. Furthermore
                  along axis <axis> the matrices should have the same number of elements. For instance,
                  valid dimensions could be the following:
                    amplitudes: [100, 2000, 100], frequencies[1, 2000, 1] for axis = 1,
                    amplitudes: [100, 100, 2000], frequencies[1, 1, 2000] for axis = 2, etc.
        Outputs:
            Normalized amplitudes. Dimensions are the same as the input amplitudes
    """
    
    df = np.diff(frequencies, axis = axis, prepend = 0.0)

    return amplitudes / np.sum(amplitudes * df, axis = axis, keepdims = True)


def _getMoments(x: np.array, y: np.array, moments: list, axis: int) -> np.array:
    """ Evaluates statistical moments of a matrix along an axis.
        Inputs:
            x      : Absicca vector or matrix  (will be broadcasted according to shape of y).
            y      : Ordinate vector or matrix (will be broadcasted according to shape of x).
            moments: Moments to evaluate, e.g. [1,2,4] for the evaluation of the first, second and fourth moment
            axis   : Axis along which these moments will be evaluated
            NOTE: x and y are assumed to have the same number of dimensions. Furthermore along axis <axis> 
                  the matrices should have the same number of elements. For instance,
                  valid dimensions could be the following:
                    y: [100, 2000, 100], x[1, 2000, 1] for axis = 1,
                    y: [100, 100, 2000], x[1, 1, 2000] for axis = 2, etc.
        Outputs: 
            Matrix of computed moments. Dimensions: x.ndims + 1. Last axes denotes the moments in ascending order.
    """

    moments = pre.expand(np.asarray(moments), x.ndim)

    # Moments vector. Dimensions: x.ndims + 1. Dimensionality of last axis= moments.shape[0]
    dx  = np.diff(x, axis = axis, prepend = 0.0)
    out = (y[..., np.newaxis] * dx[..., np.newaxis] * x[..., np.newaxis] ** moments).sum(axis = axis)

    return out


def shapeDescriptors(
    frequencies: np.array, amplitudes: np.array, axis: int, normalize: bool = True
    ) -> Tuple[np.array, np.array, np.array, np.array]:
    """ Computes several descriptors of the spectral shape.
        Inputs:
            frequencies: Vector containing the frequencies corresponding to the spectral amplitudes.
            amplitudes : Matrix containing the spectral amplitudes
            axis       : Axis along which to compute the spectral descriptors
            normalize  : Boolean indicating if the spectrum should be normalized.
        Outputs:
            Centroid  : The barycenter of the spectrum
            Spread    : Spread of the spectrum around the mean value
            Skewness  : Asymmetry of the spectrum
            Kurtosis  : Flatness of the spectrum
    """

    # Expand dimensions of the frequency vector if needed
    if frequencies.ndim != amplitudes.ndim:
        frequencies_ = pre.expand(frequencies, amplitudes.ndim, axis = axis)
    else:
        frequencies_ = frequencies

    # Normalize the spectrum
    if normalize: amplitudes = _normalizeSpectrum(frequencies_, amplitudes, axis = axis)

    # Get required spectral moments
    moments = _getMoments(frequencies_, amplitudes, moments = list(range(5)), axis = axis)

    # Compute the quantities that describe the spectral shape
    centroid = moments[..., 1] / moments[..., 0]
    spread   = np.sqrt(moments[..., 2] / moments[..., 0] - centroid ** 2)
    skewness = moments[..., 3] / moments[..., 2] ** (3/2)
    kurtosis = moments[..., 4] / moments[..., 2] ** 2

    return centroid, spread, skewness, kurtosis


def slope(x: np.array, y: np.array, axis: int) -> np.array:
    
    """ Evaluates the slope of a linear regression model on the given data in a vectorized manner.
        Inputs: 
            x    : Vector of dependent variables (arbitrary dimensions)
            y    : Matrix of independent variables of (dimensions same as x)
            axis : Axis along which to perform computations.
        Outputs:
            s : slope of linear model
    """

    # Expand dimensions of the x vector if needed
    if x.ndim != y.ndim: xEx = pre.expand(x, y.ndim, axis = axis)
    else: xEx = x

    n  = y.shape[axis]                                  # Num. points
    x_ = xEx.sum(axis = axis, keepdims = True)          # i.e.: x
    y_ = y.sum(axis = axis, keepdims = True)            # i.e.: y
    xy = (xEx * y).sum( axis = axis, keepdims = True)   # i.e.: x * y
    xx = (xEx * xEx).sum(axis = axis, keepdims = True)  # i.e.: x * x
    s  = (n * xy - x_ * y_) / (n * xx - x_ * x_)        # Fitted slope

    # Special case for a single point
    s[n == 1] = y_[n == 1]  / x_[n == 1]  

    return s


def decrease(frequencies: np.array, amplitudes: np.array, axis: int) -> np.array:
    """ Computes the gradual decrease in spectral energy as the frequency 
        increases in the frequency domain.
        Inputs: 
            frequencies: Frequency vector
            amplitudes : Matrix containing the spectral amplitudes
            axis       : Axis along which to perform computations
        Outputs:
            decrease: slope of linear model
    """

    # Expand dimensions of the frequency vector if needed
    if frequencies.ndim != amplitudes.ndim:
        frequencies_ = pre.expand(frequencies, amplitudes.ndim, axis = axis)
    else:
        frequencies_ = frequencies

    dFreq    = np.diff(frequencies_, axis = axis)
    dAmp     = np.diff(amplitudes,   axis = axis)
    decrease = np.sum(dAmp / dFreq,  axis = axis)

    return decrease


def rolloffFrequency(
    frequencies: np.array, amplitudes: np.array, axis: int, threshold: float = 0.95
    ) -> np.array:
    """ Computes the spectral roll-off point, i.e. the frequency so that 95% of the
        signal energy is contained below this frequency.
        Inputs:
            frequencies: Frequency vector
            amplitudes : Matrix containing the spectral amplitudes
            axis       : Axis along which to perform computations
            threshold  : Cut-off energy content of the signal
        Outputs:
            Roll-off frequencies matrix. Output dimensions are the same as the amplitudes input
            except from axis <axis> which contains a single element, the roll-off frequency
    """

    # Expand dimensions of the frequency vector if needed
    if frequencies.ndim != amplitudes.ndim:
        frequencies_ = pre.expand(frequencies, amplitudes.ndim, axis = axis)
    else:
        frequencies_ = frequencies

    # Ensure that all amplitudes are positive
    amps = amplitudes + np.abs(amplitudes.min(axis = axis, keepdims = True))

    # Compute the normalized cumulative sum of the power amplitudes
    ampsCumsum = np.cumsum(amps, axis = axis)
    ampsCumsum /= ampsCumsum.max(axis = axis, keepdims = True)

    # Find the index of the amplitude closest to the threshold value
    diffs = np.abs(ampsCumsum - threshold)
    ix    = np.argmin(diffs, axis = axis, keepdims = True)

    # Get roll-off frequency
    rolloffFrequency = np.take_along_axis(frequencies_, ix, axis = axis)

    return np.squeeze(rolloffFrequency, axis = axis)


def variation(amplitudes: np.array, timeAxis: int, spectralAxis: int) -> np.array:
    """ Computes the spectral variation (also known as spectral flux), i.e.e the amount of 
        variation of the spectrum along time, as the normalized cross-correlation between two
        successive amplitude spectra across time.
        Inputs:
            amplitudes  : Matrix containing the spectral amplitudes (arbitrary dimensions)
            timeAxis    : Axis containing the spectra across the time dimension
            spectralAxis: Axis containing the spectra across the frequency dimension
        Outputs:
            var: Spectral variation. Dimensions are similar to the input amplitudes, with
                the spectralAxis removed, and the timeAxis containing 1 element less.
    """

    n     = amplitudes.shape[timeAxis]
    amps0 = pre.take(amplitudes, np.arange(0, n-1), timeAxis)
    amps1 = pre.take(amplitudes, np.arange(1, n), timeAxis)
    num   = (amps0 * amps1).sum(spectralAxis)
    denom = np.sqrt( (amps0 ** 2).sum(spectralAxis) * (amps1 ** 2).sum(spectralAxis) )
    var   = 1 - num / denom

    return var


def mfcc(
    amplitudes: np.array, sampleFrequency: int, numCoefficients: int, 
    numMelFilters: int, axis: int) -> np.array:
    """ Computes the Mel-frequency cepstral coefficients (MFCCs).
        Inputs:
            amplitudes     : Matrix of complex FFT amplitudes (arbitrary dimensions)
            sampleFrequency: Sampling frequency of the signal
            numCoefficients: Number of coefficients to evaluate. Note that the first coefficient, 
                             being directly proportional to the energy is not returned.
            numMelFilters  : Number of Mel filters to be used for the conversion to Mel scale.
            axis           : Axis along which to compute the MFCCs
        Outputs:
            MFCCs: Matrix of MFCCs. Dimensions are the same as the input amplitudes, with 
                   the exception of axis <axis>, which contains <numCoefficients> elements.
    """

    # Make sure the DCT transform has enough data points
    if numCoefficients > numMelFilters: numCoefficients = numMelFilters + 1

    # Apply mid-ear filter on the input amplitudes
    _, weights = pre.filters.midEar(sampleFrequency, worN = amplitudes.shape[axis])
    weights    = pre.expand(weights, amplitudes.ndim, axis)
    amps       = np.abs(amplitudes * weights)
    
    # Mel band conversion
    melEnergy  = pre.scales.spectrogram(amps, sampleFrequency, 
        numFilters = numMelFilters, scale = 'mel', axis = axis)

    # cepstrum and MFCC
    melEnergydB = pre.powerTodb(melEnergy)
    cepstrum    = dct(melEnergydB, axis = axis, norm = "ortho")
    mfcc        = pre.take(cepstrum, np.arange(1, numCoefficients + 1), axis = axis)

    return mfcc