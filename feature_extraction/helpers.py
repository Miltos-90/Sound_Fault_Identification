import numpy as np

""" Time domain features """


def take(x: np.array, ind: np.array, axis: int) -> np.array:
    """ Retrieves slices from a given axis of a numpy array. 
        Inputs:
            x   : n-dimensional matrix from which the slices will be retrieved
            ind : Array of (integers) indices to be retrieved
            axis: Axis of <x> from which the indices will be extracted

        Outputs:
            n-dimensional matrix containing only the indices <ind> along axis <axis>
    """

    ix = [slice(None)] * x.ndim
    ix[axis] = ind
    return x[tuple(ix)]





# BELOW NOT USED YET

def rms(x: np.array, axis: int) -> np.array:
    """ Evaluates the root-mean-square (RMS) value of a signal along an axis. 
    Inputs:
        x   : Matrix containing the data
        axis: Axis along which the computations will be performed
    Outputs: 
        RMS values
    """
    return np.sqrt(np.square(x).mean(axis = axis))






def skewness(signal: np.array, axis: int):
    """ Evaluates the skewness of a signal along the given axis. """

    deAvg = signal - signal.mean(axis = axis)
    denom = (deAvg ** 2).mean(axis = axis)
    return (deAvg ** 3).mean(axis = axis) / denom ** 3/2

def kurtosis(signal: np.array, axis: int):
    """ Evaluates the kurtosis of a signal along the given axis. """

    deAvg = signal - signal.mean(axis = axis)
    denom = (deAvg ** 2).mean(axis = axis)
    return (deAvg ** 4).mean(axis = axis) / denom ** 2

def peak(signal: np.array, axis: int):
    """ Evaluates the peak value of a signal along the given axis. """
    return np.abs(signal).max(axis = axis)

def shapeFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. """
    return rms(signal, axis = axis) / np.abs(signal).mean(axis = axis)

def crestFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. """
    return signal.max(axis = axis) / rms(signal, axis = axis)

def impulseFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. """
    return peak(signal, axis = axis) / np.abs(sig).mean(axis = axis)

def clearanceFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. """
    return peak(signal, axis = axis) / np.sqrt(np.abs(sig)).mean(axis = axis) ** 2