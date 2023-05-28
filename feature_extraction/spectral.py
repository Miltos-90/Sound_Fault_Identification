import numpy as np
from typing import Tuple

def differencingAlongAxis(x: np.array, axis: int) -> np.array:
    """ Calculates the n-th discrete difference along a given axis and
        inserts zeroes on the beginning of the axis to aling dimensions
        with the input matrix.
        Inputs:
            x   : Input matrix
            axis: The axis along which the difference is taken
        Outputs:
            1-st differences. The shape of the output is the same as x.
    """
    
    newAx = [ax for ax in range(x.ndim) if ax != axis]
    dx    = np.zeros_like(x)

    np.put_along_axis(
        arr     = dx, 
        indices = np.expand_dims( np.arange(1, dx.shape[axis]), newAx), 
        values  = np.diff(x, axis = axis), 
        axis    = axis)

    return dx

def expandDimensions(arr: np.array, numDims: int, axis: int):
    """ Expands the dimensions of an array apart from a given axis.
        Inputs:
            arr     : Array whose dimensions will be expanded
            numDims : Final number of dimensions the array will have
            axis    : Axis to exclude from the expansion
        Outputs:
            Expanded array
    """

    ax = [a for a in range(numDims) if a != axis]
    return np.expand_dims(arr, ax)


def matchDimensions(
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
            frequencies : Frequency vector of the sampled amplitudes
            amplitudes  : Spectrum amplitudes
            axis        : Axis along which to normalize
        Outputs:
            Normalized amplitudes. Dimensions are the same as the input amplitudes
    """
    
    df = differencingAlongAxis(frequencies, axis = axis)

    return amplitudes / np.sum(amplitudes * df, axis = axis, keepdims = True)


def getMoments(x: np.array, y: np.array, moments: list, axis: int) -> np.array:
    """ Evaluates statistical moments of a matrix along an axis.
        Inputs:
            x      : Absicca vector or matrix  (will be broadcasted according to shape of y).
            y      : Ordinate vector or matrix (will be broadcasted according to shape of x).
            moments: Moments to evaluate, e.g. [1,2,4] for the evaluation of the first, second and fourth moment
            axis   : Axis along which these moments will be evaluated
        Outputs: 
            Matrix of computed moments. Dimensions: x.ndims + 1. Last axes denotes the moments in ascending order.
    """

    # Force numpy array
    moments = np.asarray(moments)
    
    # Expand dimensions for the moments vector: [dimensions of x + 1]
    moments = np.expand_dims(moments, list(range(x.ndim)))

    # Moments vector. Dimensions: x.ndims + 1. Dimensionality of last axis: numMoments
    dx  = differencingAlongAxis(x, axis = axis)
    out = (y[..., np.newaxis] * dx[..., np.newaxis] * x[..., np.newaxis] ** moments).sum(axis = axis)

    return out


def shapeDescriptors(
    frequencies: np.array, amplitudes: np.array, axis: int, normalize: bool = True
    ) -> Tuple[np.array, np.array, np.array, np.array]:
    """ Computes several descriptors of the spectral shape.
        Inputs:
            frequencies: Frequency vector
            amplitudes : Matrix containing the spectral amplitudes
            axis       : Axis along which to compute the spectral descriptors
            normalize  : Boolean indicating if the spectrum should be normalized.
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


def spectralSlope(x: np.array, y: np.array, axis: int) -> np.array:
    
    """ Evaluates the slope of a linear regression model on the given data in a vectorized manner.
        Inputs: 
            x    : Matrix of dependent variables (arbitrary dimensions)
            y    : Matrix of independent variables of (dimensions same as x)
            axis : Axis along which to perform computations
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


def spectralDecrease(x: np.array, y: np.array, axis: int) -> np.array:
    """ Computes the gradual decrease in spectral energy as the frequency 
        increases in the frequency domain.
        Inputs: 
            x   : Matrix of dependent variables (arbitrary dimensions)
            y   : Matrix of independent variables of (dimensions same as x)
            axis: Axis along which to perform computations
        Outputs:
            decrease: slope of linear model
    """

    dFreq    = np.diff(frequencies, axis = 1)
    dAmp     = np.diff(amplitudes, axis = 1)
    decrease = np.sum(dAmp / dFreq, axis = 1)

    return decrease