from scipy.signal.windows import hann
from scipy.fft import fft, fftfreq
from typing import Tuple
import numpy as np

""" Function used for the extraction of time-domain features """
def timeDomainFeatures(sig: np.ndarray, axis: int = 0) -> dict:
    ''' Extracts time domain features from a 2D-signal <sig> along a given axis '''

    if sig is None: raise ValueError('Empty signal detected.')
        
    rmsVal  = np.sqrt(np.square(sig).mean(axis = axis))
    sigMin  = sig.min(axis = axis)
    absSig  = np.abs(sig)
    sigMean = sig.mean(axis = axis)
    absMean = absSig.mean(axis = axis)
    peak    = absSig.max(axis = axis)
    deAvg   = sig - sigMean
    denom   = (deAvg ** 2).mean(axis = axis) # Denominator of the skewness and kurtosis calculations

    features = {
        'abs_mean' : absMean,
        'std'      : np.std(sig, axis = axis),
        'kurt'     : (deAvg ** 4).mean(axis = axis) / denom ** 2,
        'skew'     : (deAvg ** 3).mean(axis = axis) / denom ** 3/2,
        'rms'      : rmsVal,
        'peak'     : peak,
        'shape'    : rmsVal / absMean,
        'crest'    : sig.max(axis = axis) / rmsVal,
        'impulse'  : peak / absMean,
        'clearance': peak / np.sqrt(absSig).mean( axis = axis ) ** 2,
        }
        
    return features

""" Functions used for the extraction of frequency domain features. """

def _slidingWindow(sig: np.array, stride: int, overlap: float) -> np.array:
    """
    Segments a vibration signal into multiple chunks, with a given overlap.
    Inputs:
        sig:     Audio signal to be segmented [Num.samples x Num.channels]
        stride:  Stride of the signal to generate the segments [Num. samples]
        overlap: Segment overlap  [Fraction of segment stride]
    Outputs:
        out:     Segmented audio signal [Num. chunks x Num. samples of each segment x Num. channels]
    """

    if overlap >= 1 or overlap <= 0: raise ValueError('_slidingWindow(): Overlap shoule lie in (0, 1).')
    if stride <= 0: raise ValueError('_slidingWindow(): Stride should be strictly positive.')

    stride = int(stride) # Force integer

    # Get stride and offset
    noSamples, noChannels = sig.shape
    offsetSamples = int((1-overlap) * stride)

    if stride >= noSamples: 
        raise ValueError(f'_slidingWindow(): Stride should be lower than {noSamples}')

    # Compute start and end indices of each chunk, and the total number of chunks
    startIdx = np.arange(start = 0, stop = noSamples, step = offsetSamples).astype(int)
    endIdx   = np.arange(start = stride, stop = noSamples, step = offsetSamples).astype(int)
    noChunks = np.min([endIdx.shape[0], startIdx.shape[0]])

    # Make result matrix
    out = np.empty(shape = (noChunks, stride, noChannels), dtype = sig.dtype)

    for chunkNo, (start, stop) in enumerate(zip(startIdx, endIdx)):
        out[chunkNo, ...] = sig[start:stop, :]

    return out


def _detrend(sig: np.array, axis = 1) -> np.array:
    """
    Detrends a signal by subtracting its mean value along an axis.
    Inputs:
        sig: Audio signal to be detrended [Num.samples x Num.channels]
    Outputs:
        out: De-trended audio signal
    """

    mu = np.mean(sig, axis = axis)
    return sig - mu[:, None, :]


def _transformFourier(chunks: np.array, sampleFrequency: int, normalize = True) -> Tuple[np.array, np.array]:
    """ 
    Computes the Fast Fourier Transformation (FFT) of the chunked signal. 
    Inputs:
        chunks          : Chunked signal [Num. chunks x Num. samples x Num. channels]
        sampleFrequency : Sampling frequency [Hz]
        normalize       : Boolean indicating whether the FFT amplitudes should be normalized
                          (i.e. multiplied by 2.0 / Num. samples)
    Outputs:
        frequencies     : Vector of frequencies for the corresponding FFT amplitudes
        amplitudes      : FFT amplitudes [Num. chunks x Num. frequencies x Num. channels]
    """

    numSamples = chunks.shape[1] # Num samples of the audio signals

    # Make and apply window
    windowSig  = hann(M = numSamples)
    corrFactor = 1.50 # Correction factor to be applied for the Hann window
    chunks *= windowSig[None, :, None]

    # Compute (normalized) FFT amplitudes
    amplitudes  = fft(chunks, axis = 1)
    amplitudes  = np.abs(amplitudes[:, 0:numSamples//2, :])
    if normalize: amplitudes *= 2.0 / numSamples

    # Compute frequencies
    timeStep    = 1 / sampleFrequency
    frequencies = fftfreq(numSamples, timeStep)[:numSamples//2]

    # Cut-off HF
    fMax         = frequencies.max() / 1.24 # Nyquist
    idxtoKeep    = frequencies < fMax
    frequencies  = frequencies[idxtoKeep]
    amplitudes   = amplitudes[:, idxtoKeep, :]

    # Amplitude Correction due to windowing
    amplitudes *= corrFactor

    # Linear averaging of the amplitudes
    amplitudes = amplitudes.mean(axis = 0)

    return frequencies, amplitudes


def _makeOctave(band: float, limits: list = [20, 20000]):
    """ Generator of center and high/low frequency limists for octave/fractional-octave bands
        lying within a given frequency range.
        Inputs:
            band  : The octave band to compute for (1, 3/4, 1/2, 1/3, etc)
            limits: Frequency range (by default it corresponds to the audible frequency range)
        Outputs:
            frequency matrix with each row containing [low, center, high] frequencies for 
            each band (number of bands is determined according to the frequency band and the 
            user-defined frequency limits)
    """

    if band > 1 or band < 0: raise ValueError('Valid octave band range: (0, 1]')

    fLow, fHigh, fCenter = [], [], [1000.0] # Lists to hold results
    centerSpacing = 2 ** (band)             # [Hz] Spacing of center frequencies for a given octave band
    edgeSpacing   = 2 ** (1/2 *band)        # [Hz] Spacing of low(high)-to-center frequency for a given band

    while True: # Make lower half of the spectrum
        fCenter.insert(0, fCenter[0] / centerSpacing) # [Hz] Center frequency for this band
        fLow.insert(   0, fCenter[0] / edgeSpacing)   # [Hz] Min frequency for this band
        fHigh.insert(  0, fCenter[0] * edgeSpacing)   # [Hz] Max frequency for this band
        if fLow[0] <= limits[0] : break # Exit when the low frequency reaches the low-end of the acoustic spectrum

    while True: # Make upper half of the spectrum
        fLow.append(   fCenter[-1] / edgeSpacing)     # [Hz] Min frequency for this band
        fHigh.append(  fCenter[-1] * edgeSpacing)     # [Hz] Max frequency for this band
        fCenter.append(fCenter[-1] * centerSpacing)   # [Hz] Center frequency for this band
        if fHigh[-1] >= limits[1]: break # Exit when the high frequency exceeds the high-end of the acoustic spectrum

    fCenter.pop() # Remove last center frequency (not needed)

    return np.column_stack((fLow, fCenter, fHigh)) # Convert to matrix


def _makeGainCurve(fVec: np.array, fCenter: int, bandwidth: int) -> np.array:
    """ Evaluates the gain-vs-efficiency curve of a 1/b-octave filter that meets the
        Class 0 tolerance requirements of IEC 61260.
        Inputs: 
            fVec     : 1D frequency vector for which tha gains will be computed
            fCenter  : Center (middle) - band frequency of the 1/b octave filter
            bandwidth: Bandwidth designator (1 for full octave, 3 for 1/3 octave, ... etc)
        Outputs:
            g: Filter's gain
    """
    return ( 1 + ( (fVec/fCenter - fCenter/fVec) * 1.507 * bandwidth ) ** 6 ) ** (-1/2)