from .preprocessing.helpers import take
from . import preprocessing as pre
from scipy.fftpack import dct
from typing import Tuple
import numpy as np

def expandDimensions(arr: np.array, numDims: int, axis: int):
    """ Expands the dimensions of an array apart from a given axis.
        Inputs:
            arr    : Array whose dimensions will be expanded
            numDims: Final number of dimensions the array will have
            axis   : Axis to exclude from the expansion
        Outputs:
            Expanded array
    """

    ax = [a for a in range(numDims) if a != axis]
    return np.expand_dims(arr, ax)

def alignDimensions(
    arr1: np.array, arr2: np.array, axis: int) -> Tuple[np.array, np.array]:
    """ Matches the number of dimensions of two arrays, excluding a specific
        axis.
        Inputs:
            arr1, arr2: Arrays whose dimensions will be matched
            axis: Axis to exclude when adding dimensions
        Outputs: 
            Both arrays with the same number of dimensions
    """

    if   arr1.ndim < arr2.ndim: arr1 = expandDimensions(arr1, arr2.ndim, axis)
    elif arr2.ndim < arr1.ndim: arr2 = expandDimensions(arr2, arr1.ndim, axis)

    return arr1, arr2

def normalizeSpectrum(frequencies: np.array, amplitudes: np.array, axis: int) -> np.array:
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

def getMoments(x: np.array, y: np.array, moments: list, axis: int) -> np.array:
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

    # Force numpy array
    moments = np.asarray(moments)
    
    # Expand dimensions for the moments vector: [dimensions of x + 1]
    moments = np.expand_dims(moments, list(range(x.ndim)))

    # Moments vector. Dimensions: x.ndims + 1. Dimensionality of last axis: numMoments
    dx  = np.diff(x, axis = axis, prepend = 0.0)
    out = (y[..., np.newaxis] * dx[..., np.newaxis] * x[..., np.newaxis] ** moments).sum(axis = axis)

    return out

def shapeDescriptors(
    frequencies: np.array, amplitudes: np.array, axis: int, normalize: bool = True
    ) -> Tuple[np.array, np.array, np.array, np.array]:
    """ Computes several descriptors of the spectral shape.
        Inputs:
            frequencies: Matrix containing the frequencies corresponding to the spectral amplitudes
            amplitudes : Matrix containing the spectral amplitudes
            axis       : Axis along which to compute the spectral descriptors
            normalize  : Boolean indicating if the spectrum should be normalized.
            NOTE: frequencies and amplitudes are assumed to have the same number of dimensions. Furthermore
                  along axis <axis> the matrices should have the same number of elements. For instance,
                  valid dimensions could be the following:
                    amplitudes: [100, 2000, 100], frequencies[1, 2000, 1] for axis = 1,
                    amplitudes: [100, 100, 2000], frequencies[1, 1, 2000] for axis = 2, etc.
        Outputs:
            Centroid  : The barycenter of the spectrum
            Spread    : Spread of the spectrum around the mean value
            Skewness  : Asymmetry of the spectrum
            Kurtosis  : Flatness of the spectrum
    """

    # Normalize the spectrum
    if normalize: amplitudes = normalizeSpectrum(frequencies, amplitudes, axis = axis)

    # Get required spectral moments
    moments = getMoments(frequencies, amplitudes, moments = list(range(5)), axis = axis)

    # Compute the quantities that describe the spectral shape
    centroid = moments[..., 1] / moments[..., 0]
    spread   = np.sqrt(moments[..., 2] / moments[..., 0] - centroid ** 2)
    skewness = moments[..., 3] / moments[..., 2] ** (3/2)
    kurtosis = moments[..., 4] / moments[..., 2] ** 2

    return centroid, spread, skewness, kurtosis

def slope(x: np.array, y: np.array, axis: int) -> np.array:
    
    """ Evaluates the slope of a linear regression model on the given data in a vectorized manner.
        Inputs: 
            x    : Matrix of dependent variables (arbitrary dimensions)
            y    : Matrix of independent variables of (dimensions same as x)
            axis : Axis along which to perform computations
            NOTE: x and y are assumed to have the same number of dimensions. Furthermore along axis 
                  <axis> the matrices should have the same number of elements. For instance,
                  valid dimensions could be the following:
                    amplitudes: [100, 2000, 100], frequencies[1, 2000, 1] for axis = 1,
                    amplitudes: [100, 100, 2000], frequencies[1, 1, 2000] for axis = 2, etc.
        Outputs:
            s : slope of linear model
    """

    n  = y.shape[axis]                              # Num. points
    x_ = x.sum(axis = axis, keepdims = True)        # i.e.: x
    y_ = y.sum(axis = axis, keepdims = True)        # i.e.: y
    xy = (x * y).sum( axis = axis, keepdims = True) # i.e.: x * y
    xx = (x * x).sum(axis = axis, keepdims = True)  # i.e.: x * x
    s  = (n * xy - x_ * y_) / (n * xx - x_ * x_)    # Fitted slope

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
            NOTE: frequencies and amplitudes are assumed to have the same number of dimensions. Furthermore
                  along axis <axis> the matrices should have the same number of elements. For instance,
                  valid dimensions could be the following:
                    amplitudes: [100, 2000, 100], frequencies[1, 2000, 1] for axis = 1,
                    amplitudes: [100, 100, 2000], frequencies[1, 1, 2000] for axis = 2, etc.
        Outputs:
            decrease: slope of linear model
    """

    dFreq    = np.diff(frequencies, axis = 1)
    dAmp     = np.diff(amplitudes, axis = 1)
    decrease = np.sum(dAmp / dFreq, axis = 1)

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
            NOTE: frequencies and amplitudes are assumed to have the same number of dimensions. Furthermore
                  along axis <axis> the matrices should have the same number of elements. For instance,
                  valid dimensions could be the following:
                    amplitudes: [100, 2000, 100], frequencies[1, 2000, 1] for axis = 1,
                    amplitudes: [100, 100, 2000], frequencies[1, 1, 2000] for axis = 2, etc.
        Outputs:
            Roll-off frequencies matrix. Output dimensions are the same as the amplitudes input
            except from axis <axis> which contains a single element, the roll-off frequency
    """

    # Compute the normalized cumulative sum of the power amplitudes
    ampsCumsum = np.cumsum(amplitudes, axis = axis)
    ampsCumsum /= ampsCumsum.max(axis = axis, keepdims = True)

    # Find the index of the amplitude closest to the threshold value
    diffs = np.abs(ampsCumsum - threshold)
    ix    = np.argmin(diffs, axis = 1, keepdims = True)

    # Get roll-off frequency
    rolloffFrequency = np.take_along_axis(frequencies, ix, axis = axis)

    return rolloffFrequency

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
    amps0 = take(amplitudes, np.arange(0, n-1), timeAxis)
    amps1 = take(amplitudes, np.arange(1, n), timeAxis)
    num   = (amps0 * amps1).sum(spectralAxis)
    denom = np.sqrt( (amps0 ** 2).sum(spectralAxis) * (amps1 ** 2).sum(spectralAxis) )
    var   = 1 - num / denom

    return var

def mfcc(amplitudes: np.array, sampleFrequency: int, numCoefficients: int, numMelFilters: int, axis: int):
    """ Computes the Mel-frequency cepstral coefficients (MFCCs).
        Inputs:
            amplitudes     : Matrix of complex FFT amplitudes (arbitrary dimensions)
            sampleFrequency: Sampling frequency of the signal
            numCoefficients: Number of coefficients to evaluate. Note that the first coefficient, being
                              directly proportional to the energy is not returned.
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
    weights    = np.expand_dims(weights, axis = [a for a in range(amplitudes.ndim) if a != axis])
    amps       = pre.amplitudeTodb(amplitudes * weights)

    # Mel band conversion
    melEnergy  = pre.criticalBandEnergy(amps, sampleFrequency, numFilters = numMelFilters, scale = 'mel', axis = axis)

    # cepstrum and MFCC
    melEnergydB = pre.powerTodb(melEnergy)
    cepstrum    = dct(melEnergydB, axis = axis, norm = "ortho")
    mfcc        = take(cepstrum, np.arange(1, numCoefficients + 1), axis = axis)

    return mfcc