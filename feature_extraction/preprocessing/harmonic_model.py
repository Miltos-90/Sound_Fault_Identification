""" This module implements the sinusoidal harmonic model """

from scipy.signal.windows import hann
from .helpers import extract, removeSubharmonics
from typing  import Tuple
import numpy as np


def _getFundamentalFrequency(frequencies: np.array, harmonicNums: np.array, axis: int) -> np.array:
    
    """ Computes the fundamental frequency through an OLS fit of the peak frequencies found
        (dependent variable) to the peak frequencies x harmonic numbers (independent variable)
        Inputs: 
            frequencies : Matrix of peak frequencies
            harmonicNums: Corresponding harmonic numbers of the frequencies
            axis        : Axis along which to perform computations
        Outputs:
            f           : Matrix of fundamental frequencies
    """

    n  = (frequencies > 0).sum(axis = axis, keepdims = True)             # Num. points (i.e. peaks)
    x  = harmonicNums.sum(axis = axis, keepdims = True)                  # i.e.: x
    y  = frequencies.sum( axis = axis, keepdims = True)                  # i.e.: y
    xy = (harmonicNums * frequencies).sum( axis = axis, keepdims = True) # i.e.: x * y
    xx = (harmonicNums * harmonicNums).sum(axis = axis, keepdims = True) # i.e.: x * x
    f  = (n * xy - x * y) / (n * xx - x * x)    # Fitted slope = fundamental frequency

    # If a single peak was found, it corresponds to the fundamental frequency:
    f[n == 1] = y[n == 1]  

    return f


def harmonicModel(frequencies: np.array, amplitudes: np.array, peakFrequencies: np.array, numHarmonics: int, 
    axis: int, subHarmonicLimit: float = 0.75) -> Tuple[np.array, np.array]:
    """ 
        Extracts amplitudes and frequencies of the sinusoids that best approximate a signal, by 
        estimating its fundamental frequency from the spectral peaks (i.e. pitch detection) using an 
        approximate maximum likelihood detection algorithm implemented in two steps [1]:
            1. Find the peak of the histogram of the peak-frequency-differences in order to find the 
            most common harmonic spacing.
            2. Refine the above estimate using linear (OLS) regression
            3. Evaluate the slope of step 2, which gives the frequency estimate.
        
        Inputs: 
            frequencies [Hz]: 
                Vector of frequencies for the corresponding FFT amplitudes [Num. frequencies]
            amplitudes [dBFS]: Array containing FFT amplitudes [DIMS]. 
                DIMS can be any arbitrary number of dimensions, so long as the input axis <axis>
                contains <Num. frequencies> elements.
            peakFrequencies: Array of extracted peakFrequencies (output of peakFinder() function).
                This array has the same dimensions as the amplitudes, with the exception of axis <axis>
                which must contain <numHarmonics> elements.
            axis:
                Axis along which to search for the harmonic amplitudes and frequencies.
            numHarmonics: 
                Number of harmonic frequencies (multitudes of the fundamental frequency) 
                and corresponding amplitudes to extract
            subHarmonicLimit: 
                Maximum harmonic number below which extracted peaks are rejected
                (used for the removal of the DC term, subharmonics, etc.)
            
        Outputs:
            harmonicFreqs: Harmonic frequencies
            harmonicAmps : Harmonic amplitudes
            frequencies  : Peak frequencies sorted in ascending order (from lowest to highest) excluding subharmonics
                Dimensions are exactly the same as the dimensions of the input <amplitudes> matrix, with the
                exception of axis <axis>. The latter will contain <numHarmonics> elements instead of
                <Num. frequencies> elements.
            
        References: 
        [1] https://www.dsprelated.com/freebooks/sasp/Fundamental_Frequency_Estimation_Spectral.html
    """
    
    freqs, harmonics = removeSubharmonics(peakFrequencies, subHarmonicLimit, axis)
    fundamentalFreq  = _getFundamentalFrequency(freqs, harmonics, axis)
    harmonicFreqs    = np.concatenate([fundamentalFreq * (i + 1) for i in range(numHarmonics)], axis = axis)
    harmonicAmps     = extract(x = frequencies, y = amplitudes, xq = harmonicFreqs, axis = axis)
    
    return harmonicFreqs, harmonicAmps

