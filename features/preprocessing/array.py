""" Collection of function to manipulate the raw audio signals prior to further preprocessing. """

import numpy as np

def pad(x: np.array, padValue, padSize:int, padType: str, axis: int) -> np.array:
    """ Pads an array with a given value according to a given size and type
        along an axis.
        Inputs:
            x       : Array to be padded
            padValue: Value to pad the array with
            padSize : Number of elements to be inserted to the array
            padType : Type of padding to be used. Can be one of:
                * 'start' : Values will be inserted at the start of the axis
                * 'end'   : Values will be inserted at the end of the axis
                * 'center': Array will be 'centered', i.e. hald the values will
                            be inserted at the start, and the other half at the end
            axis    : Axis along which the above operation will be performed.
        Outputs:
            Padded array
    """

    # Split padding accordingly
    if padSize > 0 and padType != 'none': # padding needed

        if padType == 'end' or padType == 'start':

            # Make a single array that will be used to pad the input array
            arr = _makePadArray(axis, padSize, padValue, x.shape)
            
            # Prepare iterable of arrays to be padded
            if   padType == 'start': arrays = (arr, x)
            elif padType == 'end'  : arrays = (x, arr)

        elif padType == 'center':

            if padSize % 2 != 0: # Odd number of elements need to be added. Split unevenly
                padStart, padEnd = padSize // 2, padSize // 2 + 1

            else: # Even number of elements need to be added. Even split
                padStart, padEnd = [padSize // 2] * 2
            
            # Prepare iterable of arrays
            arrStart = _makePadArray(axis, padStart, padValue, x.shape)
            arrEnd   = _makePadArray(axis, padEnd,   padValue, x.shape)
            arrays   = (arrStart, x, arrEnd)

        x = np.concatenate(arrays, axis = axis, casting = 'safe')

    return x

def chunk(signal: np.array, frameSize: int, hopSize: int, axis: int) -> np.array:
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

def getPadSize(numElements: int, frameSize: int, hopSize: int) -> int:
    """ Computes the padding size needed to split an array along an axis
        of <numElements> elements into <frameSize> frames with a stride of
        <hopSize> elements.
        Inputs:
            numElements: Number of elements of the array to be padded 
                         along the axes that will be padded
            frameSize  : Number of elements in each frame
            hopSize    : Stride to be used for the frames
        Outputs:
            padSize    : Number of additional elements that are needed for
                         the array to contain the exact number of frames
    """

    numFrames = np.ceil( (numElements - frameSize) / hopSize + 1)
    padSize   = int((numFrames - 1) * hopSize + frameSize) - numElements
    return padSize

def _makePadArray(axis: int, padSize: int, padValue, shape: tuple) -> np.array:
    """ Generates an array of <padValue> that will be used to pad an array of
        size <shape> along axis <axis>.
    """

    shape       = list(shape) # Force mutable
    shape[axis] = int(padSize)
    return padValue * np.ones(shape = shape, dtype = type(padValue))