""" This module implements a Bark scale filterbank """

import numpy as np
from numpy.fft  import rfftfreq


def _hertz2bark(frequency: np.array) -> np.array:
    """
    Convert a given frequency vector from Hertz to the Bark scale.
    Inputs:
        frequency: The frequency in Hertz.
    Outputs:
        The corresponding value(s) on the Bark scale.
    """
    return 6 * np.arcsinh(frequency / 600)

def _bark2hertz(bark: np.array) -> np.array:
    """
    Convert a given frequency vector from Bark scale to Hertz.
    Inputs:
        frequency: The frequency in Hertz.
    Outputs:
        The corresponding value(s) on the Bark scale.
    """
    return 600 * np.sinh(bark / 6)

def _makeFilters(rangeVector: np.array, centersVector: np.array) -> np.array:
    """ Returns a Bark filter bank across a frequency range and for various frequency vectors.
        Inputs:
            rangeVector   : Frequency vector for the filter design
            centersVectros: Frequency vector containing the center frequencies for each filter
            NOTE: Both to be given in the Bark scale
        Outputs:
            out: Matrix containing the filter bank with dimensions [len(rangeVector), len(centersVector)]
    """

    rangeM, centerM = np.meshgrid(rangeVector, centersVector, indexing='ij')
    deltas = rangeM - centerM       # Range - center
    out    = np.zeros_like(deltas) # Output matrix

    # Compute cases
    c1 = (deltas < -1.3) | (2.5 < deltas)
    c2 = (-1.3 <= deltas) & (deltas <= -0.5)
    c3 = (-0.5 < deltas) & (deltas < 0.5)
    c4 = ~ (c1 | c2 | c3)

    # Fill in output matrix
    out[c1] = 0
    out[c3] = 1
    out[c2] = 10 ** (2.5 * (deltas[c2] + 0.5))
    out[c4] = 10 ** (-1 * (deltas[c4] - 0.5))

    return out

def filterbank(
    numDFT: int, numFilters: int, sampleFrequency: int, normalize: bool = True
    ) -> np.array:
    """ Computes the filters in a Bark filterbank and return the corresponding
        transformation matrix.
        Inputs:
            numDFTbins     : The number of DFT bins
            numFilters     : The number of mel filters to include in the filterbank
            sampleFrequency: The sample rate/frequency for the signal
            normalize      : Whether to scale the Bark filter weights by their area in Mel space

        Outputs:
            fbank      : The barj-filterbank transformation matrix. Columns correspond to filters,
                          rows to DFT bins. Dimensions: [numDFTbins // 2 + 1, numFilters]
            barkCenters: Frequency centers [Hz] for the FFT bins. Dimensions: [numDFTbins // 2 + 1]
        
    """

    minFreq, maxFreq = 0, sampleFrequency // 2
    minBark, maxBark = _hertz2bark(minFreq), _hertz2bark(maxFreq)

    # uniformly spaced values on the Bark scale
    barkCenters = np.linspace(minBark, maxBark, numFilters)

    # FFT frequencies
    frequencies = rfftfreq(numDFT, 1.0 / sampleFrequency)

    # Make Filter bank
    fBank = _makeFilters(_hertz2bark(frequencies), barkCenters)

    # Normalize
    if normalize:
        barkBins   = _bark2hertz(np.linspace(minBark, maxBark, numFilters+2))
        energyNorm = 2.0 / (barkBins[2 : numFilters+2] - barkBins[:numFilters])
        fBank     *= energyNorm[np.newaxis, :]

    return fBank[1:,:]