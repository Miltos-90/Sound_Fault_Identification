import numpy as np
from numpy.fft import rfftfreq
from typing import Tuple

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


def melFilterbank(
    numDFTbins: int, numFilters: int, sampleFrequency: int, normalize: bool = True) -> Tuple[np.array, np.array]:
    """
    Compute the filters in a Mel filterbank and return the corresponding
    transformation matrix. This implementation is based on code in the LibROSA 
    package [1] and is taken from the numpy-ml github repository [2]
    
    Inputs:
        numDFTbins     : The number of DFT bins
        numFilters     : The number of mel filters to include in the filterbank
        sampleFrequency: The sample rate/frequency for the signal
        normalize      : Whether to scale the Mel filter weights by their area in Mel space

    Outputs:
        fbank : The mel-filterbank transformation matrix. Columns correspond to filters,
                rows to DFT bins. Dimensions: [numDFTbins // 2 + 1, numFilters]
        hzBins: Frequency centers [Hz] for the FFT bins. Dimensions: [numDFTbins // 2 + 1]
    
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

    return melBins, fBank.T[1:, :]


def makeEinsumNotation(matrix1Dims: int, matrix2Dims: int, axis: int) -> str:
    """ Dynamic generation of the einsum notation string based on the 
        dimensions of the input matrices and the axis of the first matrix
        that will be multipled with the weights of the Mel filterbank.
        Inputs:
            matrix1Dims: Number of dimensions of the first matrix
            matrix2Dims: Number of dimensions of the second matrix
            axis       : Axis of the first matrix that will be operated upon
        Outputs:
            notation: Einsum notation string
    """

    def makeMatrixStr(nDims: int, alphabet: list) -> list:
        """ Generates the subscript notation for a <nDims>-dimensional matrix. """

        ix, sub = 0, []
        while ix < nDims and alphabet:
            # Grab the first available letter of the (remaining) 
            # alphabet and assign it to the next dimension.
            sub.append(alphabet.pop(0))
            ix += 1

        return sub

    alphabet = list('abcdefghijklmnopqrstuvwxyz')

    # Check matrix shape
    maxLen = len(alphabet) - 2
    if matrix1Dims > maxLen: # Reserve one letter for the weights matrix
        raise RuntimeError(
            f'Input matrix should have lower than {maxLen} dimensions.')


    # Make LHS
    # Make strings for the two matrices
    str1 = makeMatrixStr(matrix1Dims, alphabet)
    str2 = makeMatrixStr(matrix2Dims, alphabet)

    # Assign the same subscript to the axes that will be multiplied
    commonLetter = str1[axis]
    str2[0]      = commonLetter

    # Convert to strings and make left-hand-side of the einsum notation string
    str1 = ''.join(str1)
    str2 = ''.join(str2)
    lhs  = ', '.join([str1, str2])

    # Make right-hand-side:
    # Make a sorted list with all unique subscripts for both matrices,
    # and remove the subscript that will be summed upon (the commoLetter
    # in both matrices)
    rhs = list(set(str1 + str2))
    rhs.sort()
    rhs = ''.join(rhs)

    # Merge LHS and RHS
    notation = ' -> '.join([lhs, rhs])

    return notation


def criticalEnergy(weightedSpectrum: np.array, freqToMelMap: np.array, axis: int) -> np.array:
    """ Computes the critical badn energy of the Mel scale. 
        Inputs: 
            weightedSpectrum: Power spectrum pre-multiplied by the filter wieghts.
                              Dimensions: [..., numBands]
            freqToMelMap    : Frequency to Mel band index mapper
            axis            : Axis of the weighted spectrum along which to compute the critical energy
        Outputs:
            out: Matrix containing the critical energy of the corresponding Mel bands.
                 The output dimensions correspond to the dimensions of the input matrix, with 
                 <numBands> elements along axis <axis>
    """

    # Make output matrix
    numBands = weightedSpectrum.shape[-1]
    sh       = list(weightedSpectrum.shape)[: -1] # Exclude number of bands here
    sh[1]    = numBands
    out      = np.empty(shape = sh)

    # Expand vector x to match the shape of the output matrix
    axExpand = [ax for ax in range(out.ndim) if ax != axis]
    expand   = lambda x: np.expand_dims(np.asarray(x), axExpand)

    for filterNo in range(numBands):

        # Grab the indices of the frequencies that belong to the current Mel band
        filterIx = np.where(freqToMelMap - 1 == filterNo)[0]

        # Sum the corresponding elements of the Mel-weighted power spectrum
        melEnergy = np.take_along_axis(
            weightedSpectrum[..., filterNo], indices= expand(filterIx), axis = axis
            ).sum(axis = axis, keepdims = True)
        
        # Place them along the filter axis of the output matrix
        np.put_along_axis(out, expand([filterNo]), melEnergy, axis = axis)

    return out


def energy(
    frequencies: np.array, amplitudes: np.array, sampleFrequency: int, numBands: int, axis: int
    ) -> Tuple[np.array, np.array]:
    """ Computes the critical band energy in the Mel scale.
        Inputs:
            frequencies    : Vector of frequencies of the power spectrum
            amplitudes     : Matrix of the amplitudes of the power spectrum
            sampleFrequency: Sampling frequency of the signal
            numBands       : Number of Mel bands
            axis           : Axis along which to compute the critical energy
        Outputs:
            out: Critical energy of the Mel bands
    """

    numFFT   = frequencies.shape[0] * 2
    bins, w  = melFilterbank(numFFT, numBands, sampleFrequency)
    indices  = np.digitize(frequencies, bins)
    notation = makeEinsumNotation(amplitudes.ndim, w.ndim, axis = axis)
    energies = np.einsum(notation, amplitudes, w)
    out      = criticalEnergy(energies, indices, axis)

    return out