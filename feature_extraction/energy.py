""" This module contains all functions that extract the energy-related features of a sound signal. """
import numpy as np

def energy(x: np.array, axis: int) -> np.array:
    """ Evaluates the energies of the signals contained in an axis of the input matrix in the frequency domain. 
    Inputs:
        x   : Matrix containing the spectral amplitudes along one dimension
        axis: Axis along which the computations will be performed
    Outputs: 
        Energies of the signals along the requested axis
    """
    return np.abs(x ** 2).sum(axis = axis)
