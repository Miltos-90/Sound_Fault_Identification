""" Collection of filtering-related functions. """

from scipy.signal import hilbert, firwin2, freqz
from scipy.fft    import fft
from typing       import Tuple
import matplotlib.pyplot as plt
import numpy as np



def _load() -> np.array:
    """ Loads the relevant data from [1], [2] for the design of the mid ear filter. """

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

def _transform(data: np.array, fs: int) -> Tuple[np.array, np.array]:
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

def _toMinPhaseFilter(coeffs: np.array) -> np.array:
    """ Converts the  filter coefficients of the FIR filter (output of the firwin2 function) 
        to the corresponding coefficients of a minimum phase filter. 
    """

    X      = np.fft.fft(coeffs)
    Xmin   = np.abs(X) * np.exp( -1j * np.imag( hilbert( np.log( np.abs(X) ) ) ) )
    coeffs = np.real(np.fft.ifft(Xmin))

    return coeffs

def _plotResponse(freqs: np.array, response: np.array):
    """ Plots the response of the filter """

    plt.figure(figsize = (6, 4))
    plt.semilogx(freqs, todb(response, 1))
    plt.title('Middle ear filter frequency response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB re 20uPa]')
    plt.show()

    return


def midEar(sampleFrequency: int, plot: bool = False, **kwargs) -> Tuple[np.array, np.array]:
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

    order = 511                                      # IIR Filter order
    data  = _load()
    freq, ampl = _transform(data, sampleFrequency)   # Load and process filter data
    b     = firwin2(order, freq, ampl)               # Design filter
    b     = _toMinPhaseFilter(b)                     # To minimum phase
    w, h  = freqz(b, fs = sampleFrequency, **kwargs) # Compute frequency response
    if plot: _plotResponse(w, h)                     # Plot frequency response

    return w, h

def Aweighting(frequencies: np.array, dbMin: float = None) -> np.array:
    """ 
    Computes the A-weighting (https://en.wikipedia.org/wiki/A-weighting) of a set of frequencies.
    Inputs:
        frequencies : One or more frequencies ot be converted [Hz]
        dbMin       : Clip weights below this threshold [dB]. If set to None, no clipping is performed.
    Outputs:
        weights     : Weighting matrix  [dB]

    This implementation is taken from the excellent librosa library:
        https://github.com/librosa/librosa/blob/main/librosa/core/convert.py#L1847
    """

    fSquare = np.asanyarray(frequencies) ** 2
    const   = np.array( [12194.217, 20.598997, 107.65265, 737.86223] ) ** 2
    weights = np.array(2.0 + 20.0 * (
        np.log10(const[0])
        + 2 * np.log10(fSquare)
        - np.log10(fSquare + const[0])
        - np.log10(fSquare + const[1])
        - 0.5 * np.log10(fSquare + const[2])
        - 0.5 * np.log10(fSquare + const[3])
    ))

    if dbMin is None: return weights
    else: return np.maximum(min_db, weights)