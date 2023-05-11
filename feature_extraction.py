import numpy as np

""" Time domain features """

def rms(signal: np.array, axis: int):
    """ Evaluates the root-mean-square (RMS) value of a signal along the given axis. """
    return np.sqrt(np.square(signal).mean(axis = axis))

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