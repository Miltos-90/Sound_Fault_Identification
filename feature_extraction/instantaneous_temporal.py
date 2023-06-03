""" This module contains all functions that extract the instantaneous temporal features of a sound signal. """

import numpy as np
from .preprocessing.helpers import take

def makeEinsumNotation(numDims: int, axis: int) -> str:
    """ Generates the einsum notation for the inner product of
        two matrices of equal dimensions along an axis.
        Inputs:
            numDims: Number of dimensions of the matrices (equal for both)
            axis   : Axis along which the inner product will be computed
        Outputs:
            String containing the notation for np.einsum to compute the inner product
    """

    alphabet = list('abcdefghijklmnopqrstuvwxyz')

    # Check matrix shape
    maxLen = len(alphabet) - 2
    if numDims > maxLen: # Reserve one letter for the weights matrix
        raise RuntimeError(f'Input matrix should have lower than {maxLen} dimensions.')

    # Make subscripts for the dimensions of the input matrix
    ix, s = 0, []
    while ix < numDims and alphabet:
        # Grab the first available letter of the alphabet 
        # and assign it to the next dimension.
        s.append(alphabet.pop(0))
        ix += 1

    s = ''.join(s)

    # Make left hand side of the notation
    lhs = ', '.join([s, s])

    # The right hand side is simply all other dimension(s) of the matrix
    rhs = ''.join([c for i, c in enumerate(s) if i != axis])

    notation = ' -> '.join([lhs, rhs])
    
    return notation

def autocorrelation(x: np.array, numLags: int, axis: int, center: bool = False) -> np.array:
    """ Computes the autocorrelation of the signals along one dimension of a matrix, and
        returns the autocorrelation coefficients.
        Inputs:
            x      : Matrix containing the data.
            numLags: Number of lags to return autocorrelation for. If not provided,
                     uses min(10 * np.log10(nobs), nobs - 1). The returned value
                     includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1).
            axis:    Axis containing the signals (time-series)
            center:  Flag indicating whether or not to subtract the average of each time-series
                     from the data.
        Outputs:
            acf:     Matrix with estimated autocorrelations. The dimensions of the output matrix equal those of
                     the input matrix with the exception of axis <axis>, which contains <numLags> elements.

    """

    # Check num. lags
    n = x.shape[axis]
    
    if numLags is None: 
        lagLen = min(10 * np.log10(n), n - 1)
    elif numLags > n - 1: 
        raise ValueError(f"Number of lags should be lower than the signal length {n}.")
    else:
        lagLen = numLags + 1

    # Center signal if needed
    if center: x -= x.mean(axis, keepdims = True)

    # Generator of slices from the signal along an axis given the input indices
    slicer = lambda index: take(x, index, axis)

    # Initialize autocovariance matrix:
    # It's shape is equal to the shape of the input matrix apart from 
    # axis <axis>, which contains lagLen + 1 elements.
    covShape = [size for dimNo, size in enumerate(x.shape) if dimNo != axis]
    covShape.append(lagLen + 1)
    acov     = np.empty(shape = covShape)

    # Compute inner product along the <axis>-th dimension
    notation     = makeEinsumNotation(x.ndim, axis = axis)
    acov[..., 0] = np.einsum(notation, x, x) 


    for i in range(lagLen):
        ix1 = np.arange(i + 1, n)
        ix2 = np.arange(0, n - (i + 1))
        acov[..., i + 1] = np.einsum(notation, slicer(ix1), slicer(ix2))

    acov /= n # Normalize

    # Compute autocorrelation function
    acf = acov / np.expand_dims(acov[..., 0], -1)

    # Swapaxes to match input matrix dimensions
    acf = np.swapaxes(acf, axis, -1)

    return take(acf, np.arange(1, lagLen), axis = axis)

def zeroCrossingRate(x: np.array, axis: int) -> np.array:
    """ Computes the zero-crossing rate (i.e. number of sign changes) along one dimension of an arbitrarily shaped matrix.
    Inputs:
    x   : Matrix containing the data
    axis: Axis along which the computations will be performed
    Outputs:
        Matrix with the same dimensions as <x> excluding the <axis> dimension.
    """

    return np.sum(np.diff(np.sign(x) >= 0, axis = axis), axis = axis)