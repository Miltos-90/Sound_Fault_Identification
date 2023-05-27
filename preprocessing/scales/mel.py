""" This module implements a Mel scale filterbank """

import numpy as np
from numpy.fft  import rfftfreq

def hertz2mel(frequency: np.array) -> np.array:
    """
    Convert frequency from the Hertz scale to the Mel scale.
    Inputs:
        frequency: Frequency Vector [Hz]
    Outputs:
        mel: Frequency value in Mel scale
    """

    return 2595 * np.log10(1 + (frequency / 700))

def mel2hertz(mel: np.array) -> np.array:
    """
    Convert Mels to Hertz scale
    Inputs:
        mel: Frequency value in Mel scale
    Outputs:
        frequency: Frequency Vector [Hz]
    """

    return  700 * (10 ** (mel / 2595) - 1)

def filterbank(numDFTbins: int, numFilters: int, sampleFrequency: int, normalize: bool = True) -> np.array:
    """
    Computes the filters in a Mel filterbank and return the corresponding
    transformation matrix. This implementation is based on code in the LibROSA 
    package [1] and is taken from the numpy-ml github repository [2]
    
    Inputs:
        numDFTbins     : The number of DFT bins
        numFilters     : The number of mel filters to include in the filterbank
        sampleFrequency: The sample rate/frequency for the signal
        normalize      : Whether to scale the Mel filter weights by their area in Mel space

    Outputs:
        fbank  : The mel-filterbank transformation matrix. Columns correspond to filters,
                 rows to DFT bins. Dimensions: [numDFTbins // 2 + 1, numFilters]
    
    References: 
    [1] McFee et al. (2015). "librosa: Audio and music signal analysis in
            Python", Proceedings of the 14th Python in Science Conference.
            https://librosa.github.io
    [2]  Numpy-ml github repository: https://github.com/ddbourgin/numpy-ml/tree/master

    """
    minFreq, maxFreq = 0, sampleFrequency // 2
    minMel, maxMel   = hertz2mel(minFreq), hertz2mel(maxFreq)
    
    # uniformly spaced values on the mel scale, translated back into Hz
    melBins = mel2hertz(np.linspace(minMel, maxMel, numFilters + 2))

    # the centers of the frequency bins for the DFT
    hzBins  = rfftfreq(numDFTbins, 1.0 / sampleFrequency)
    melDiff = np.diff(melBins)
    fBank   = np.zeros((numFilters, numDFTbins // 2 + 1))
    ramps   = melBins.reshape(-1, 1) - hzBins.reshape(1, -1)

    for i in range(numFilters):

        # Calculate the filter values on the left and right across the bins
        left  = -ramps[i] / melDiff[i]
        right = ramps[i + 2] / melDiff[i + 1]

        # Set them zero when they cross the x-axis
        fBank[i] = np.maximum(0, np.minimum(left, right))

    if normalize:
        energyNorm = 2.0 / (melBins[2 : numFilters + 2] - melBins[:numFilters])
        fBank     *= energyNorm[:, np.newaxis]

    return fBank.T[1:, :]