""" Collection of functions related to envelope analysis. """

from scipy.signal import hilbert
from .array import chunk
import numpy as np


def amplitude(signal: np.array, axis: int):
    """
    Computes the amplitude envelope of a signal using the Hilbert transform.
    Inputs:
        signal: The input signal.
        axis  : Axis along which the envelope will be computed
    Outputs:
        The envelope of the input signal.
    """
    analyticSignal = hilbert(signal, axis = axis)
    return np.abs(analyticSignal)

def energy(signal, frameSize: int, hopSize: int, axis):
    """
    Computes the energy (RMS) envelope of a signal.
    Inputs:
        signal   : The input signal
        frameSize: The size of each frame.
        hopSize  : The hop size between adjacent frames.
        axis     : Axis along which the envelope will be computed
    Outputs:
        The envelope of the input signal.
    """

    def rms(signal: np.array, axis: int):
        """ Evaluates the root-mean-square (RMS) value of a signal along the given axis. """
        return np.sqrt(np.square(signal).mean(axis = axis))
    
    frames = chunk(signal, frameSize = frameSize, hopSize = hopSize, axis = axis)
    return rms(frames, axis = axis + 1)
