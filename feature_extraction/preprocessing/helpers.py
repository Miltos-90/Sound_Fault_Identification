import numpy as np
from typing import Union


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


def makeSlice(numDims: int, indices: np.array, axis: int) -> tuple:
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
    
    return x[makeSlice(x.ndim, indices = indices, axis = axis)]
