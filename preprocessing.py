"""
This module implements the preprocessing steps mentioned in Section 2 of:

"A large set of audio features for sound description (similarity and classification)
in the CUIDADO project", Peeters G., 2004.

URL: http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
(accessed 11/05/2023)
"""

from scipy.signal.windows import hann
from scipy.signal import hilbert, firwin2, freqz
from scipy.fft import fft, fftfreq
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

# =================== Time-series preprocessing functions =====================
def sliceArray(signal: np.array, frameSize: int, hopSize: int, axis = -1) -> np.array:
    """
    Slices an array into overlapping (or non-overlapping) frames along a specified axis, 
    using stride manipulation

    Inputs:
        signal   : The input data array.
        frameSize: The size of each frame.
        hopSize  : The hop size between adjacent frames.
        axis     : The axis along which to slice the data array. Defaults to the last axis

    Outputs:
        The sliced data array. The 'frame' axis precedes the one specified as input
    """

    signalShape   = list(signal.shape)
    signalStrides = list(signal.strides)

    # Compute the shape and strides for the new sliced array
    newShape = signalShape.copy()
    newShape[axis] = (signalShape[axis] - frameSize) // hopSize + 1
    newShape.insert(axis + 1, frameSize)

    newStrides = signalStrides.copy()
    newStrides[axis] *= hopSize
    newStrides.insert(axis + 1, signalStrides[axis])

    # Create a view of the original data array with the new shape and strides
    return np.lib.stride_tricks.as_strided(signal, shape = newShape, strides = newStrides)

def detrend(signal: np.array, axis: int) -> np.array:
    """
    Detrends a signal by subtracting its mean value along an axis.
    Inputs:
        signal: Signal to be detrended
        axis: Axis alogn which to detrend.
    Outputs:
        out: De-trended signal
    """

    mu = np.mean(signal, axis = axis)
    return signal - np.expand_dims(mu, axis = axis)

# =================== Frequency spectra related functions =====================
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

def fftSpectrum(signal: np.array, sampleFrequency: int, axis: int) -> Tuple[np.array, np.array]:
    """ 
    Computes the Fast Fourier Transformation (FFT) and Power Spectral Density (PSD_) of the chunked signal. 
    Inputs:
        signal          : Chunked signal [Num. frames x Num. samples x Num. channels]
        sampleFrequency : Sampling frequency [Hz]
        axis            : Axis along which to apply the FFT
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
    freqs = fftfreq(numSamples, 1 / sampleFrequency)
    sigF  = fft(signal, axis = axis, norm = "ortho")                
    Sxx   = sigF * sigF.conj() / sampleFrequency

    # Apply window corrections
    acf, ecf = _windowCorrectionFactors(windowSig)
    sigF *= acf
    Sxx  *= ecf

    # Cut-off at the highest frequency and ignre negative frequencies
    fNyquist  = sampleFrequency / 2.0
    keep      = np.where( (freqs >= 0) & (freqs < fNyquist) )[0]
    freqs     = freqs[keep]
    sigF      = sigF.take(keep, axis = axis)
    Sxx       = Sxx.take(keep, axis = axis)

    return freqs, sigF, Sxx

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
        octavePSD[band, :] = (psdAmplitudes.real[bandIndex, :] * gainCurve[:, None] ** 2).sum(axis = 0)

    # Make spectrum
    octaveNoiseLevels       = np.sqrt(octavePSD)
    octaveCenterFrequencies = octaveFrequencyRange[:, 1]

    return octaveCenterFrequencies, octaveNoiseLevels

def todb(array: np.array, reference = 1.0):
    """ Convert an array to decibels [dB], relative to the specified reference level.
        Inputs:
            array: Array to convert to [dB]
            reference: The reference level [in the units of the array]
        Outputs: 
            The array translated to [dB]

    """
    return 20 * np.log10(np.abs(array) / reference)

# =================== Envelope analysis related functions =====================
def amplitudeEnvelope(signal: np.array, axis: int):
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

def energyEnvelope(signal, frameSize: int, hopSize: int, axis = 1):
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

    def rms(signal: np.array, axis: int):
        """ Evaluates the root-mean-square (RMS) value of a signal along the given axis. """
        return np.sqrt(np.square(signal).mean(axis = axis))
    
    frames = sliceArray(signal, frameSize = frameSize, hopSize = hopSize, axis = axis)
    return rms(frames, axis = axis + 1)

# =========================== Additional functions =============================
def sinusoidalHarmonicModel(frequencies: np.array, amplitudes: np.array, harmonics: int, axis: int):
    """ Extracts (complex) amplitudes and frequencies of the sinusoids that best approximate a signal, by 
        analyzing the signal's spectrum and identifying the frequencies and amplitued of its spectral peaks.
        Inputs: 
            frequencies: Vector of frequencies for the corresponding FFT amplitudes
            amplitudes : FFT amplitudes [Num. frames x Num. frequencies x Num. channels]
            harmonics  : Number of harmonic frequencies and amplitudes to extract
            axis       : Axis along which to search for the harmonic amplitudes and frequencies
        Outputs:
            harmonicFreqs: Harmonic frequencies [Num. frames x harmonics x Num. channels]
            harmonicAmps:  Harmonic amplitudes  [Num. frames x harmonics x Num. channels]
    """

    # Compute the fundamental frequency
    peakIndex   = np.argmax(np.abs(amplitudes), axis = axis)
    fundamental = frequencies[peakIndex]

    # Compute sinusoidal harmonic frequencies and corresponding amplitudes
    frames        = amplitudes.shape[0]
    channels      = amplitudes.shape[2]
    shape         = (frames, harmonics, channels)
    harmonicFreqs = np.empty(shape = shape, dtype = frequencies.dtype)
    harmonicAmps  = np.empty(shape = shape, dtype = amplitudes.dtype)
    freqExpanded  = np.expand_dims(frequencies, axis = tuple(range(1, amplitudes.ndim)))

    for hNo in range(1, harmonics + 1):

        # Get the spectrum frequency that is closest to the current harmonic frequency
        diff = freqExpanded - fundamental * hNo
        idx  = np.abs(diff).argmin(axis = 0)
        harmonicFreqs[:, hNo-1, :] = frequencies[idx]

        # get the corresponding amplitude
        idx     = np.expand_dims(idx, axis = axis)
        curAmps = np.take_along_axis(amplitudes, idx, axis = axis)
        harmonicAmps[:, hNo-1, :] = np.squeeze(curAmps, axis = axis)
        
    return harmonicFreqs, harmonicAmps

def middleEarFilter(sampleFrequency: int, plot: bool = False, **kwargs) -> Tuple[np.array, np.array]:
    """ 
    Computes the filter coefficients of a FIR filter approximating thr effect of the middle ear according to [1], 
    using a part of thedata published in [2]. This code is based on the MIDDLEEARFILTER function of the Auditory 
    Modelling Toolbox for Matlab/Octave [3].
    Inputs:
        sampleFrequency: Sampling frequency of the digital system [Hz]
        plot: Whether or not to plot the frequency response of the filter
        kwargs: Additional arguments for the scipy.signal.freqz function [4]
    Outputs:
        w: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the 
            range [0, pi) (radians/sample).
        h: The frequency response, as an array of complex numbers.

    References:
        [1] E. Lopez-Poveda and R. Meddis. A human nonlinear cochlear filterbank. J. Acoust. Soc. Am., 110:3107--3118, 2001.
        [2] R. Goode, M. Killion, K. Nakamura, and S. Nishihara. New knowledge about the function of the human middle 
            ear: development of an improved analog model. The American journal of otology, 15(2):145--154, 1994.
        [3] URL: http://amtoolbox.org/ (last accessed: May 2023)
        [4] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html (last accessed: May 2023)
    """

    def load() -> np.array:
        """ Loads the relevant data from [1], [2] for the design of the filter. """

        data = np.array([ # Non-extrapolated points (Fig. 1 of [2])
            [400,  0.19953], [600,  0.22909], [800,  0.21878], [1000, 0.15136],
            [1200, 0.10000], [1400, 0.07943], [1600, 0.05754], [1800, 0.04365],
            [2000, 0.03311], [2200, 0.02754], [2400, 0.02188], [2600, 0.01820],
            [2800, 0.01445], [3000, 0.01259], [3500, 0.00900], [4000, 0.00700],
            [4500, 0.00457], [5000, 0.00500], [5500, 0.00400], [6000, 0.00300],
            [6500, 0.00275]])

        # get velocity (proportional voltage) acc. to formula in [2]
        data[:,1] = data[:,1] * 2 * np.pi * 1e-6 * data[:,0]

        # to get data at 0dB SPL (assumed that stapes velocity is linearly related to pressure)
        data[:,1] = data[:,1] * 10 ** (-104/20)

        # to get stapes PEAK velocity, multiply amplitudes by sqrt(2)
        data[:,1] = data[:,1] * np.sqrt(2)

        extrp = np.array([ # Extrapolated points (Fig. 2b of [1])
            [100, 1.181e-09],  [200, 2.363e-09],  [7000, 8.705e-10],
            [7500, 8.000e-10], [8000, 7.577e-10], [8500, 7.168e-10],
            [9000, 6.781e-10], [9500, 6.240e-10], [10000, 6.000e-10]])

        # Concatenate
        mask1  = extrp[:, 0] < data[0, 0]
        mask2  = extrp[:, 0] > data[-1, 0]
        extrp1 = extrp[mask1]
        extrp2 = extrp[mask2]
        data   = np.concatenate([extrp1, data, extrp2], axis=0)

        return data

    def transform(data: np.array, fs: int) -> Tuple[np.array, np.array]:
        """ Processes the data according to the sampling frequency <fs> [Hz] and transforms it 
            in a suitable format for the scipy.signal.firwin2 function.
            """

        fs2 = fs / 2
        if fs <= 20000:
            # Cut the table because the sampling frequency is too low to accomodate the full range.
            indx = np.where(data[:, 0] < fs2)[0]
            data = data[:indx[-1] + 1, :]
        else:
            # otherwise the table will be extrapolated towards fs/2
            # data point added every 1000Hz
            lgth = data.shape[0]
            for _ in range(1, int((fs2 - data[-1, 0])/1000) + 1):
                data = np.vstack([data, [data[-1, 0] + 1000, data[-1, 1] / 1.1]])
            
        # for the function firwin2 the last data point has to be at fs/2
        lgth = data.shape[0]
        if data[-1, 0] != fs2:
            data = np.vstack([
                data, 
                [fs2, data[-1, 1] / (1 + (fs2 - data[-1, 0]) * 0.1/1000)]
                ])
            
        # Extract the frequencies and amplitudes, and put them in the format that firwin2 likes.
        freq = np.hstack([0, data[:, 0] * (2/fs)]) # Frequency [Hz]
        ampl = np.hstack([0, data[:, 1]])          # Stapes peak velocity [m/s]

        return freq, ampl / ampl.max() # Frequency [Hz], Amplitude [-]

    def toMinPhaseFilter(coeffs: np.array) -> np.array:
        """ Converts the  filter coefficients of the FIR filter (output of the firwin2 function) 
            to the corresponding coefficients of a minimum phase filter. 
        """

        X      = np.fft.fft(coeffs)
        Xmin   = np.abs(X) * np.exp( -1j * np.imag( hilbert( np.log( np.abs(X) ) ) ) )
        coeffs = np.real(np.fft.ifft(Xmin))

        return coeffs

    def plotResponse(freqs: np.array, response: np.array):
        """ Plots the response of the filter """

        plt.figure(figsize = (6, 4))
        plt.semilogx(freqs, todb(response, 1))
        plt.title('Middle ear filter frequency response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude [dB re 20uPa]')
        plt.show()

        return

    order = 511                                      # IIR Filter order
    data  = load()
    freq, ampl = transform(data, sampleFrequency)    # Load and process filter data
    b     = firwin2(order, freq, ampl)               # Design filter
    b     = toMinPhaseFilter(b)                      # To minimum phase
    w, h  = freqz(b, fs = sampleFrequency, **kwargs) # Compute frequency response
    if plot: plotResponse(w, h)                      # Plot frequency response

    return w, h