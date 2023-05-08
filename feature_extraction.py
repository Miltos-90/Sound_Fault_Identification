from scipy.signal.windows import hann
from scipy.fft import fft, fftfreq
from typing import Tuple
import numpy as np

"""
This module implements the feature extraction process in the time- and frequency- domains.
It is largely based on Section 4.4 of [1]

References:
[1] Niu, Gang. "Data-driven technology for engineering systems health management." Springer Singapore 10 (2017): 978-981.

"""

""" Functions used for the extraction of time-domain features """
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

""" Fourier transformation """

def _windowCorrectionFactors(windowSignal: np.array) -> Tuple[float, float]:
    """ Computes the amplitude and energy correction factors for a window signal according to [1]. 
        (agrees with the values reported in [2] for various windows.)
        Inputs: window signal vector
        Outputs:
            (scalar) amplitude correction factor
            (scalar) energy correction factor

        References:
        [1] Brandt, A., 2011. Noise and vibration analysis: signal analysis and experimental procedures. John Wiley & Sons.
        [2] https://community.sw.siemens.com/s/article/window-correction-factors
    """

    n   = windowSignal.shape[0]                  # Signal length
    acf = n / np.sum(windowSignal)               # Amplitude Correction Factor (Spectrum, Autopower, and Orders)
    ecf = np.sqrt(n / np.sum(windowSignal ** 2)) # Energy Correction Factor (Power Spectral Density)

    return acf, ecf


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


def _fftAndPower(chunks: np.array, sampleFrequency: int) -> Tuple[np.array, np.array]:
    """ 
    Computes the Fast Fourier Transformation (FFT) and Power Spectral Density of the chunked signal. 
    Inputs:
        chunks          : Chunked signal [Num. chunks x Num. samples x Num. channels]
        sampleFrequency : Sampling frequency [Hz]
        normalize       : Boolean indicating whether the FFT amplitudes should be normalized
                          (i.e. multiplied by 2.0 / Num. samples)
    Outputs:
        frequencies     : Vector of frequencies for the corresponding FFT amplitudes
        amplitudes      : FFT amplitudes [Num. chunks x Num. frequencies x Num. channels]
    """

    # Num samples of each chunk
    numSamples = chunks.shape[1]

    # Make and apply window
    windowSig  = hann(M = numSamples)
    chunks *= windowSig[None, :, None]

    # Compute FFT amplitudes and power spectral density
    sigF = fft(chunks, axis = 1)                
    Sxx  = sigF * sigF.conj() / sampleFrequency
    sigF = np.abs(sigF)
    Sxx  = Sxx.real

    # Ignore negative frequencies
    Sxx  = Sxx[:, :numSamples//2, :] 
    sigF = sigF[:, :numSamples//2, :]

    # Normalize
    sigF *= 2.0 / numSamples
    Sxx  *= 2.0 / numSamples

    # Apply window corrections
    acf, ecf = _windowCorrectionFactors(windowSig)
    sigF *= acf
    Sxx  *= ecf

    # Compute frequencies
    timeStep    = 1 / sampleFrequency
    frequencies = fftfreq(numSamples, timeStep)[:numSamples//2]

    # Cut-off at the highest frequency
    fNyquist     = 1.0 / timeStep / 2.0 # Nyquist
    idxtoKeep    = frequencies < fNyquist

    frequencies  = frequencies[idxtoKeep]
    sigF         = sigF[:, idxtoKeep, :]
    Sxx          = Sxx[:, idxtoKeep, :]

    # Linear averaging
    sigF = sigF.mean(axis = 0)
    Sxx  = Sxx.mean(axis = 0)

    return frequencies, sigF, Sxx


def spectra(sig: np.array, stride: int, overlap: int, frequency: int) -> Tuple[np.array, np.array]:
    """ Preprocesses the signal and performs an FFT 
        Inputs:
            sig:        Audio signal to be segmented [Num.samples x Num. channels]
            stride:     Stride of the signal to generate the segments [Num. samples]
            overlap:    Segment overlap  [Fraction of segment stride]
            frequency:  Sampling frequency [Hz] of the input signal <sig>
        Outputs:
            amplitudes: Amplitudes of one-dimensional n-point discrete Fourier Transform (DFT) applied
                        along the last axis (challen dimension) of the input signal <sig>, with dime-
                        nsions [n-points for FFT, Num. channels]
            frequencies: Frequency vector of the corresponding amplitudes in [Hz] with dimensions [n-points for FFT]
    """

    chunks = _slidingWindow(sig, stride, overlap)
    chunks = _detrend(chunks)
    frequencies, fftLevels, powerLevels = _fftAndPower(chunks, frequency)

    return frequencies, fftLevels, powerLevels


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


def octaveSpectrum(
    frequencies: np.array, psdAmplitudes: np.array, bandwidthDesignator: int, referenceNoiseLevel: float
    ) -> Tuple[np.array, np.array]:
    """ Computes a 1/b octave spectrum from the FFT amplitudes nand frequencies (i.e. the output of
        the FFT function).
        Inputs:
            frequencies:         Vector [Hz] designating the frequencies corresponding to the FFT amplitudes [num FFT points].
            psdAmplitudes:       Power Spectral Density (psd) amplitudes [num. FFT points, num. channels]
            bandwidthDesignator: (1 for full octave, 3 for 1/3 octave, ... etc)
            referenceNoiseLevel: Reference level [dB] for the conversion of the amplitudes to [dB]
        Outputs:
            octaveCenterFrequencies: Frequency vector (x-axis) of the spectrum [Hz]
            octaveNoiseLevels:       Noise matrix (y-axis) of the spectrum [num. frequncies, num.channels]
    """
    
    # Get frequency bands for the given octave
    octaveFrequencyRange = _makeOctave(band = 1 / bandwidthDesignator, limits = [20, frequencies.max()])
    
    # Matrix with PSD levels for each octave band [Num. bands, Num. channels]
    octavePSD = np.zeros(shape = (octaveFrequencyRange.shape[0], psdAmplitudes.shape[1]))

    # Compute PSD levels for each band
    for band, (fLow, fCenter, fHigh) in enumerate(octaveFrequencyRange):

        bandIndex          = (frequencies >= fLow ) & (frequencies <= fHigh )
        bandFrequencies    = frequencies[bandIndex]
        gainCurve          = _makeGainCurve(bandFrequencies, fCenter, bandwidthDesignator)
        octavePSD[band, :] = (psdAmplitudes[bandIndex, :] * gainCurve[:, None] ** 2).sum(axis = 0)

    # Make spectrum
    octaveRMS               = np.sqrt(octavePSD)
    octaveNoiseLevels       = 20 * np.log10(octaveRMS / referenceNoiseLevel)
    octaveCenterFrequencies = octaveFrequencyRange[:, 1]

    return octaveCenterFrequencies, octaveNoiseLevels