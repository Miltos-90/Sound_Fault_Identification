""" Collection of function to manipulate the raw audio signals prior to further preprocessing. """


import numpy as np

def chunk(signal: np.array, frameSize: int, hopSize: int, axis = -1) -> np.array:
    """
    Slices an array into overlapping (or non-overlapping) frames along a specified axis, 
    using stride manipulation

    Inputs:
        signal   : The input data array.
        frameSize: The size of each frame.
        hopSize  : The hop size between adjacent frames.
        axis     : The axis along which to slice the data array. Defaults to the last axis

    Outputs:
        The sliced data array. The 'frame' axis precedes the one specified as input
    """

    signalShape   = list(signal.shape)
    signalStrides = list(signal.strides)

    # Compute the shape and strides for the new sliced array
    newShape = signalShape.copy()
    newShape[axis] = (signalShape[axis] - frameSize) // hopSize + 1
    newShape.insert(axis + 1, frameSize)

    newStrides = signalStrides.copy()
    newStrides[axis] *= hopSize
    newStrides.insert(axis + 1, signalStrides[axis])

    # Create a view of the original data array with the new shape and strides
    return np.lib.stride_tricks.as_strided(signal, shape = newShape, strides = newStrides)

def detrend(signal: np.array, axis: int) -> np.array:
    """
    Detrends a signal by subtracting its mean value along an axis.
    Inputs:
        signal: Signal to be detrended
        axis: Axis alogn which to detrend.
    Outputs:
        out: De-trended signal
    """

    mu = np.mean(signal, axis = axis)
    return signal - np.expand_dims(mu, axis = axis)
