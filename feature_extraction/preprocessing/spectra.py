""" Collection of functions used to perform frquency-space analysis of the signals. """
from scipy.signal.windows import hann
from scipy.fft import fft, fftfreq
from typing import Tuple
import numpy as np

def windowCorrectionFactors(windowSignal: np.array) -> Tuple[float, float]:
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


def fourier(signal: np.array, sampleFrequency: int, axis: int, **kwargs) -> Tuple[np.array, np.array]:
    """ 
    Computes the Fast Fourier Transformation (FFT) of a signal along an axis
    Inputs:
        signal          : n-Dimensional array for the FFT computation
        sampleFrequency : Sampling frequency [Hz]
        axis            : Axis along which to apply the FFT
        kwargs          : Additional arguments for the scipy.fft or scipy.fftfreq functions
    Outputs:
        frequencies     : Vector of frequencies for the corresponding FFT amplitudes
        amplitudes      : FFT amplitudes on the above frequencies
    """

    # Num samples of each chunk
    numSamples = signal.shape[axis]

    # Make and apply window
    windowSig  = hann(M = numSamples)
    axesExpand = list(range(signal.ndim)) # Axes along which to expand the window
    axesExpand.remove(axis)
    signal *= np.expand_dims(windowSig, axesExpand)

    # Compute FFT and power spectral density
    n     = kwargs.pop("n", numSamples)
    freqs = fftfreq(n, 1 / sampleFrequency)
    norm  = kwargs.pop("norm", "ortho") # ortho is the default for the 'norm' argument
    sigF  = fft(signal, **kwargs, axis = axis, norm = "ortho", n = n)

    # Apply window corrections
    acf, _ = windowCorrectionFactors(windowSig)
    sigF *= acf
    
    # Cut-off at the highest frequency and ignore negative frequencies
    fNyquist  = sampleFrequency / 2.0
    keep      = np.where( (freqs >= 0) & (freqs < fNyquist) )[0]
    freqs     = freqs[keep]
    sigF      = sigF.take(keep, axis = axis)

    return freqs, sigF


def psd(spectrum: np.array, sampleFrequency: int) -> np.array:
    """ Computes the power spectral density (PSD) from the FFT spectrum. 
        Inputs:
            signal          : n-Dimensional array for the FFT computation
            sampleFrequency : Sampling frequency [Hz]
        Outputs:
            amplitudes: PSD amplitudes
    """

    _, ecf = windowCorrectionFactors(hann(M = spectrum.shape[0]))
    Sxx = spectrum * spectrum.conj() / sampleFrequency

    return Sxx.real * ecf


def makeOctave(band: float, limits: list = [20, 20000]):
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


def octave(frequencies: np.array, amplitudes: np.array, bandwidthDesignator: int, axis: int) -> Tuple[np.array, np.array]:
    """ Computes a 1/b octave spectrum from the FFT amplitudes nand frequencies (i.e. the output of
        the FFT function).
        Inputs:
            frequencies         : Vector designating the frequencies corresponding to the FFT amplitudes [Hz].
            amplitudes          : Power Spectral Density (psd) amplitudes
            bandwidthDesignator : (1 for full octave, 3 for 1/3 octave, ... etc)
            axis                :  Axis of the psd amplitudes along which to compute the spectrum
        Outputs:
            octaveCenterFrequencies : Frequency vector (x-axis) of the spectrum [Hz]
            octaveNoiseLevels       : Noise matrix (y-axis) of the spectrum [num. frequncies, num.channels]
    """

    # Get frequency bands for the given octave
    octaveBand      = 1/bandwidthDesignator
    frequencyLimits = [20, frequencies.max()]
    frequencyRange  = makeOctave(octaveBand, frequencyLimits)

    # Matrix with PSD levels for each octave band
    # Matrix shape: Along the axis on which the octave calculation will be 
    # performed the size is equal to the size of the frequency vector.
    # Along the axis on which the octave calculation will be performed
    # the size is equal to the size of the frequency vector
    shape       = list(amplitudes.shape)
    shape[axis] = frequencyRange.shape[0]
    octavePSD   = np.zeros(shape = shape)

    # Compute PSD levels for each band

    # The dimensions of various quantities in the loop below will need to be 
    # expanded to match the dimensions of the input in all other axis apart
    # from the one given as an input (the one over which the octave levels are
    # being evaluated). The below two lines define a convenient wrapper to 
    # expand the dimensions on-the-fly.
    axes2expand = [dim for dim in range(amplitudes.ndim) if dim != axis]
    expand      = lambda arr: np.expand_dims(arr, axis = axes2expand)

    for bandNo, (fLow, fCenter, fHigh) in enumerate(frequencyRange):

        # Get the narrowband frequencies corresponding to this octave
        bandIdx   = np.where( (frequencies >= fLow ) & (frequencies <= fHigh ) )[0]
        bandFreqs = frequencies[bandIdx]

        # Compute the gain-efficiency curve
        gain = _makeGainCurve(bandFreqs, fCenter, bandwidthDesignator)
        
        # Grab the PSD amplitudes corresponding to this band
        curPSD = np.take_along_axis(amplitudes, expand(bandIdx), axis = axis)

        # Compute amplitude for this octave
        octaveAmpl = (curPSD * expand(gain ** 2)).sum(axis = axis, keepdims = True)

        # Put octave levels in the matrix
        np.put_along_axis(octavePSD, indices = expand([bandNo]), values = octaveAmpl, axis = axis)

    # Make spectrum
    noiseLevels       = np.sqrt(octavePSD)
    centerFrequencies = frequencyRange[:, 1]

    return centerFrequencies, noiseLevels


def powerTodb(array: np.array, reference = 1.0):
    """ Converts spectral power to decibels [dB], relative to the specified reference level.
        Inputs:
            array    : Power array to convert to [dB]
            reference: The reference level [in the units of the array]
        Outputs: 
            The array translated to [dB]
    """

    if np.iscomplexobj(array): array = np.abs(array)
    return 10 * np.log10(array / reference)


def amplitudeTodb(array: np.array, reference = 1.0):
    """ Convert spectral amplitudes to decibels [dB], relative to the specified reference level.
        Inputs:
            array    : Amplitude array to convert to [dB]
            reference: The reference level [in the units of the array]
        Outputs: 
            The array translated to [dB]
    """

    return powerTodb(np.square(array), reference**2)
