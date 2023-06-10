import numpy as np
from typing import Union, Tuple


def extract(x: np.array, y: np.array, xq: np.array, axis: int) -> np.array:
    """ Extracts specific elements of array y at the requested points x along a given axis.
        Inputs:
            x   : Array of x-coordinates at which y has been sampled. Vector of <n> elements
            y   : Array of y-coordinates. Matrix of arbitrary dimensions with axis <axis> containing <n> elements
            xq  : Array of x-coordinates for which y is needed. Matrix of arbitrary dimensions, which should match 
                  the dimensions of matrix y on all axes apart from axis <axis>.
            axis: Axis along which the elements will be extracted. y matrix should have <n> elements along this axis.
        Outputs:
            yq  : Array of y-coordinates. Matrix with dimensions equal to xq

        NOTE 1: This function does not perform interpolation. It returns the values of y for the indices 
                of the values of x that are *closest* to the values xq.
        NOTE 2:
        Example of valid input dimensions (for 3d matrices) should conform to the following notation:
            x:  (n,)
            y:  (i, n, k)
            xq: (i, j, k)
        which will produce output
            yq: (i, j, k)
    """

    # Match/expand dimensions of the x vector to allow for vectorized
    # searching.
    x_ = _vectorToMatrix(x, numDims = y.ndim, axis = axis)
    yq = np.empty_like(xq, dtype = y.dtype)

    # Get corresponding eleents of the y matrix
    for h in range(xq.shape[axis]):

        # Extract y elements corresponding to this  number
        xqCur   = take(xq, indices = np.array([h]), axis = axis)
        closeIx = np.argmin(np.abs(x_ - xqCur), axis = axis, keepdims = True)
        yqCur   = np.take_along_axis(y, closeIx, axis = axis)

        # Add to results
        ix = _makeSlice(yq.ndim, np.array([h]),  axis)
        yq[ix] = yqCur

    return yq


def _vectorToMatrix(vector: np.array, numDims: int, axis: int) -> np.array:
    """Converts a vector to matrix of appropriate dimensions.
        Inputs: 
            vector : Vector to be converted to a matrix
            numDims: Number of dimensions of the output matrix
            axis   : Axis along which the elements of the vector will be placed
        Outputs:
            mtrx   : Matrix with <numDims> dimensions, with the elements of the 
                     input vector lying across the <axis>-th axis.
    """

    addAxes  = tuple([ax for ax in range(numDims - 1)])
    mtrx     = np.expand_dims(vector, axis = addAxes)
    mtrx     = mtrx.swapaxes(-1, axis) 

    return mtrx


def expand(arr: np.array, numDims: int, axis: int = None):
    """ Expands the dimensions of an array apart from a given axis.
        Inputs:
            arr    : Array whose dimensions will be expanded
            numDims: Final number of dimensions the array will have
            axis   : Axis to exclude from the expansion. If None, all
                     axes are included in the array
        Outputs:
            Expanded array
    """

    if axis is not None:
        ax = [a for a in range(numDims) if a != axis]
    else:
        ax = list(range(numDims))
    
    return np.expand_dims(arr, ax)


def _makeSlice(numDims: int, indices: np.array, axis: int) -> tuple:
    """ Generates indices to slice an array along a dynamically specified axis
        Inputs:
            numDims: Number of dimensions of the matrix to be sliced
            indices: Indices along an axis to be sliced from the matrix
            axis   : Axis along which to slice the matrix
        Outputs:
            ix: Indices that slice the matrix accordingly when using mtrx[ix]
    """
    ix       = [slice(None)] * numDims
    ix[axis] = indices

    return tuple(ix)


def take(x: np.array, indices: Union[int, np.array], axis: int) -> np.array:
    """ Retrieves slices from a given axis of a numpy array. 
        Inputs:
            x      : n-dimensional matrix from which the slices will be retrieved
            indices: Array of (integers) indices to be retrieved
            axis   : Axis of <x> from which the indices will be extracted

        Outputs:
            n-dimensional matrix containing only the indices <ind> along axis <axis>
    """
    
    return x[_makeSlice(x.ndim, indices = indices, axis = axis)]
    #return np.take(x, indices, axis)


def _medianFrequencyDiff(x: np.array, axis: int) -> np.array:
    """ Computes the median of the 1st-order difference of a matrix along a given axis.
        Inputs:
            x   : Matrix for which the quantities will be computed
            axis: Axis along which the quantity will be computed
        Outputs:
            medianDiff: Median of the 1st-order difference
    """

    spacing    = np.diff(x, axis = axis)
    masked     = np.where(spacing != 0, spacing, np.nan)
    medianDiff = np.nanmedian(masked, axis = axis)

    # In case a single peak is found, the median spacing is equal to the frequency of that peak
    numPeaks   = (x > 0).sum(axis = axis)
    singlePeak = numPeaks <= 1

    # Compute the mean frequency among all the non-zero entries. For a single peak, this operation
    # defaults to the frequency of the single peak.
    masked     = np.where(x != 0, x, np.nan)
    mean_      = np.nanmean(masked, axis = axis)

    # Convert to arrays (in case a single sample has been passed as an input)
    mean_      = np.asarray(mean_)
    medianDiff = np.asarray(medianDiff)
    medianDiff[singlePeak] = mean_[singlePeak]

    return np.expand_dims(medianDiff, axis = axis)


def removeSubharmonics(frequencies: np.array, subLimit: float, axis: int
    ) -> Tuple[np.array, np.array]:
    """ Deletes subharmonics, dc peak, etc. that may have been picked up by the peak 
        finding algorithm.
        Inputs:
            amplitudes : FFT levels
            frequencies: Frequencies of the FFT levels used as inputs
            subLimit   : Harmonic limit below which peaks are discarded
        Outputs:
            frequencies : Frequencies with subharmonics removed
            harmonicNums: Harmonic numbers
    """
    
    medianDiff   = _medianFrequencyDiff(frequencies, axis)
    harmonicNums = np.round(frequencies / medianDiff.astype(float))
    sAxes        = tuple([ax for ax in range(frequencies.ndim) if ax != axis])
    subharms     = np.where(~(harmonicNums < subLimit).all(axis = sAxes))[0]

    # Remove them
    harmonicNums = np.take(harmonicNums, subharms, axis)
    frequencies  = np.take(frequencies, subharms, axis)

    return frequencies, harmonicNums