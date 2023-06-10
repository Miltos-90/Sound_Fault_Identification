""" Collection of functions related to envelope analysis. """

from scipy.signal import hilbert
from scipy.fft import fft, ifft
from .helpers import expand
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


def _rms(signal: np.array, axis: int):
    """ Evaluates the root-mean-square (RMS) value of a signal along the given axis. """
    return np.sqrt(np.square(signal).mean(axis = axis))


def energy(signal, frameSize: int, hopSize: int, axis: int):
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
    
    frames = chunk(signal, frameSize = frameSize, hopSize = hopSize, axis = axis)

    return _rms(frames, axis = axis + 1)


def _makeIndexMatrix(shape: tuple, axis: int) -> np.array:
    """ Generates a multidimensional array of index numbers.
        Inputs:
            shape: Shape of the array to be generated
            axis: Axis along which the indices will be evaluated
        Outputs:
            ind: Matrix containing the indices
        Example input/output:
            >>> _makeIndexMatrix(shape = (2, 3), axis = 1)
            [0, 1, 2],
            [0, 1, 2]
    """

    shape = list(shape) # Force list

    # Convert to matrix
    ind = expand(arr = np.arange(shape[axis]), numDims = len(shape), axis = axis)

    # Repeat along all other axes
    shape[axis] = 1
    ind  = np.tile(ind, reps = shape)

    return ind


def _makeLifter(shape: tuple, axis: int, sampleFrequency: np.array, fundamentalFrequency: np.array) -> np.array:
    """ Generates a rectangular cepstrum filter (lifter).
        Inputs:
            shape                : Shape of the cepstrum matrix being liftered
            axis                 : Axis along which the lifter will be applied
            sampleFrequency      : Sampling rate of the signals in Hertz
            fundamentalFrequency : Fundamental frequencies of the signals in Hertz
        Outuputs:
            lift: Lifter matrix
    """

    shape = list(shape)
    nFFT  = shape[axis]

    # Number of elements along axis <axis> that will be windowed
    period = np.round(sampleFrequency / fundamentalFrequency).astype(int)
    nw     = np.asarray(2 * (period - 2))  # Almost 1 period left and one period right
    nw[nw % 2 == 0] += 1                   # Make the even numbers odd

    # Get n-dimensional region that will be filled with zeros
    nw     = np.expand_dims(nw, axis = 0)
    ind    = _makeIndexMatrix(shape, axis)
    zeroes = ind >  nw // 2 + 1 | (ind < nFFT - nw // 2) # Zero-region of the cepstrum filter (lifter)

    # Make array
    lift = np.ones(shape = shape)
    np.putmask(lift, mask = zeroes, values = 0)

    return lift


def spectral(amplitudes: np.array, sampleFrequency: int, fundamentalFrequency: np.array, axis: int) -> np.array:
    """ Computes the spectral envelope using cepstral windowing.
        Inputs:
            amplitudes          : Matrix containing the spectral amplitudes, of arbitraty dimensions
            sampleFrequency     : The sampling frequency of the signals
            fundamentalFrequency: Matrix containing the fundamental frequencies of the signals
                                  represented in the amplitudes matrix
            axis                : Axis along which to perform the computations
        Outputs:
            envelope: Spectral envelopes of the input amplitudes. Dimensions match the dimensions
                      of the <amplitudes> matrix
    """

    shape     = amplitudes.shape
    cepstrum  = ifft(amplitudes, axis = axis).real # real cepstrum

    # window the cepstrum (apply lifter)
    cepstrum *= _makeLifter(shape, axis, sampleFrequency, fundamentalFrequency)  

    envelope  = fft(cepstrum, axis = axis).real # spectral envelope (real part)

    return envelope