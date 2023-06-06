""" This module contains all functions that extract the features of a sound signal 
    in the time-domain.
"""

import numpy as np
from typing import Union

def rms(signal: np.array, axis: int) -> np.array:
    """ Evaluates the root-mean-square (RMS) value of a signal along an axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            RMS values
    """
    return np.sqrt(np.square(signal).mean(axis = axis))

def power(signal: np.array, axis: int):
    """ Evaluates the power of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Power array
    """
    return np.square(signal).sum(axis = axis) / signal.shape[axis]

def skewness(signal: np.array, axis: int, norm: bool = False):
    """ Evaluates the skewness of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Skewness array
    """

    if norm: deAvg = signal - signal.mean(axis = axis, keepdims = True)
    else   : deAvg = signal
    
    denom = (deAvg ** 2).mean(axis = axis)
    return (deAvg ** 3).mean(axis = axis) / denom ** 3/2

def kurtosis(signal: np.array, axis: int, norm: bool = False):
    """ Evaluates the kurtosis of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Kurtosis array
    """

    if norm: deAvg = signal - signal.mean(axis = axis, keepdims = True)
    else   : deAvg = signal

    denom = (deAvg ** 2).mean(axis = axis)
    return (deAvg ** 4).mean(axis = axis) / denom ** 2

def peak(signal: np.array, axis: int):
    """ Evaluates the peak value of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Peak values array
    """
    return np.abs(signal).max(axis = axis)

def peak2peak(signal: np.array, axis: int):
    """ Evaluates the peak-to-peak value of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Peak-to-peak array
    """
    return signal.max(axis = axis) - signal.min(axis = axis)

def shapeFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Shape factors array
    """
    return rms(signal, axis = axis) / np.abs(signal).mean(axis = axis)

def crestFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Crest factors array
    """
    return signal.max(axis = axis) / rms(signal, axis = axis)

def impulseFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Impulse factor array
    """
    return peak(signal, axis = axis) / np.abs(signal).mean(axis = axis)

def clearanceFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Clearance factors array
    """
    return peak(signal, axis = axis) / np.sqrt(np.abs(signal)).mean(axis = axis) ** 2