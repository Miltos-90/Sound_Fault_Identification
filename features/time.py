""" This module contains all functions that extract the features of a sound signal 
    in the time-domain.
"""

import numpy as np
from typing import Union


def features(signal: np.array, axis: int) -> np.array:
    """ Extracts all features in the time-domain from an array of signals.
        Inputs:
            signal: Array containing signals in the time-domain for which the 
                    features will be extracted.
            axis:   Axis along which the signals (time-series) are arranged over time.
        Outputs:
            features: Array of features extracted in the time-domain. The output array
                has the same dimensions as the input signal array, with the exception of
                axis <axis>, which contains 13 elements (i.e. features).
    """

    features = np.stack([
        np.min(signal, axis = axis),
        np.max(signal, axis = axis),
        np.std(signal, axis = axis),
        _rms(             signal, axis = axis),
        _power(           signal, axis = axis),
        _skewness(        signal, axis = axis),
        _kurtosis(        signal, axis = axis),
        _peak(            signal, axis = axis),
        _shapeFactor(     signal, axis = axis),
        _crestFactor(     signal, axis = axis),
        _impulseFactor(   signal, axis = axis),
        _clearanceFactor( signal, axis = axis),
        _peak2peak(       signal, axis = axis)
        ], axis = axis)

    return features


def _rms(signal: np.array, axis: int) -> np.array:
    """ Evaluates the root-mean-square (RMS) value of a signal along an axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            RMS values
    """
    return np.sqrt(np.square(signal).mean(axis = axis))


def _power(signal: np.array, axis: int):
    """ Evaluates the power of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Power array
    """
    return np.square(signal).sum(axis = axis) / signal.shape[axis]


def _skewness(signal: np.array, axis: int, norm: bool = False):
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


def _kurtosis(signal: np.array, axis: int, norm: bool = False):
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


def _peak(signal: np.array, axis: int):
    """ Evaluates the peak value of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Peak values array
    """
    return np.abs(signal).max(axis = axis)


def _peak2peak(signal: np.array, axis: int):
    """ Evaluates the peak-to-peak value of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Peak-to-peak array
    """
    return signal.max(axis = axis) - signal.min(axis = axis)


def _shapeFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Shape factors array
    """
    return _rms(signal, axis = axis) / np.abs(signal).mean(axis = axis)


def _crestFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Crest factors array
    """
    return signal.max(axis = axis) / _rms(signal, axis = axis)


def _impulseFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Impulse factor array
    """
    return _peak(signal, axis = axis) / np.abs(signal).mean(axis = axis)


def _clearanceFactor(signal: np.array, axis: int):
    """ Evaluates the crest factor of a signal along the given axis. 
        Inputs:
            signal: Array containing the signals
            axis  : Axis along which the computations will be performed
        Outputs: 
            Clearance factors array
    """
    return _peak(signal, axis = axis) / np.sqrt(np.abs(signal)).mean(axis = axis) ** 2