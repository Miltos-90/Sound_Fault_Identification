""" This module contains all functions that extract the energy-related features of a sound signal. """

from .helpers import rms

def totalEnergy(x: np.array, axis: int) -> np.array:
    """ Evaluates the total energies of the signals contained in an axis of the input matrix. 
    Inputs:
        x   : Matrix containing the time-series signals along one dimension
        axis: Axis along which the computations will be performed
    Outputs: 
        Total energies of the signals along the requested axis
    """
    return rms(x, axis = axis) ** 2


def harmonicEnergy(x: np.array, axis: int) -> np.array:
    """ Evaluates the harmonic energies of the signals contained in an axis of the input matrix. 
    Inputs:
        x   : Matrix containing the harmonic spectral amplitudes along one dimension
        axis: Axis along which the computations will be performed
    Outputs: 
        Harmonic energies of the signals along the requested axis
    Notes: The harmonic spectral amplitudes can be evaluated from the harmonicModel() function of 
           the preprocessing module, with the raw (complex) FFT amplitudes as its inputs.
    """
    return np.abs(x).sum(axis = axis)