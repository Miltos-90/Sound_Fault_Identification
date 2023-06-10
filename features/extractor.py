""" This module contains the main function that extracts all the features from 
    the raw audio signals.
"""

import numpy as np
from typing import Literal
from . import preprocessing as pre
from . import temporal, spectral, harmonic, perceptual, time, various, temporal_global


def makeFeatures(
    signal: np.array, fs: int, 
    frameSize   : int = 2 ** 12, hopSize   : int = 2 ** 10, 
    envFrameSize: int = 2 ** 9,  envHopSize: int = 2 ** 6, 
    padType     : Literal["center", "start", "end", "center", "none"] = 'center', 
    padValue    : float = 0.0, 
    numFilters  : int = 24, harmonics: int = 10, 
    numMFCC     : int = 12, numLags  : int = 12, 
    octaveDesignator: int = 3, axis: int = 0) -> np.array:
    """
    Feature extraction from the raw audio signals [1]. The raw signals are split
    into consecutive time-frames, and after the necessary pre-processing the
    following features are extracted for each frame:

    Time domain features
   ===============================
    1. Minimum value
    2. Maximum value
    3. Standard deviation
    4. Root Mean Squared value
    5. Power
    6. Skewness
    7. Kurtosis
    8. Peak value
    9. Shape factor
    10. Crest factor
    11. Clearance factor
    12. Peak-to-peak value
    ===============================
    
    Instantaneous temporal features
    ===============================
    1. Zero-crossing rate
    2. Autocorrelation coefficients
    ===============================

    Spectral features
    ===============================
    1. Shape descriptors:
        * Spectral centroid
        * Spectral spread
        * Spectral skewness
        * Spectral kurtosis
    2. Spectrum slope
    3. Spectral decrease
    4. Roll-off frequency
    5. Spectral variation
    6. Spectral variation average (over time)
    7. Spectral variation variance (over time)
    8. Maximum amplitude
    9. Mel-frequency cepstral coefficients (MFCC)
    10. 1st derivative of MFCC (over time)
    11. 2nd derivative of MFCC (over time)
    12. Peak frequencies
    13. Peak relative amplitudes (1st peak is omitted - normalized spectrum)
    14. Amplitudes in 1/3 octave band
    ===============================

    Perceptual features
    ===============================
    1. Total loudness
    2. Relative specific loudness (Bark bands)
    3. Acoustic spread
    4. Acoustic sharpness
    ===============================

    Harmonic features
    ===============================
    1. Harmonic spectral deviation
    2. Inharmonicity
    3. Noisiness
    4. Odd-to-even energy ratio
    5. Tristimulus
    6. Fundamental frequency
    ===============================

    Various features (that do not fit in the other categories)
    ===============================
    1. Tonality coefficient
    ===============================

    Apart from the full set of features being extracted on the time-domain signals,
    a subset of the spectral features is also extracted from the Bark spectrogram,
    and the instantaneous temporal features are extracted from the amplitude
    and energy envelopes. 

    Subsequently, the weighted average of those features over time
    is evaluated, with the weights corresponding to the instantaneous loudness
    of the signal over time.

    Inputs:
        signal      : Array of signals to extract features from (arbitrary dimensions)
        fs          : Sampling frequency [Hz]
        frameSize   : Size of the frames that the signal is split into
        hopSize     : Hop size of the adjacent frames the signal is split into
        envFrameSize: Size of the frames that the signal's RMS envelope is split into
        envHopSize  : Hop of the frames that the signal's RMS envelope is split into
        padType     : Type of padding to be applied in order to split the signal
                      in frames of the input frame- and hop- size.
                      Possible values:
                        "center", "start", "end", "center", "none"
        padValue    : Value to pad the signals with if needed
        numFilters  : Number of Bark bands to be used for the Bark spectrogram
        harmonics   : Number of harmonic frequencies to extract
        numMFCC     : Number of Mel Frequency Cepstral Coefficients to extract
        numLags     : Number of autocorrelation lags to extract
        octaveDesignator: Designator of the octave band whose spectrum will be included
                          as a features (i.e. =3 for 1/3 octave band, etc..)
        axis        : Axis along which the signals are arranged over time
    Outputs:
        features    : Array of extracted features. It's dimensions equal the dimensions
                      of the input signal array with the excpetion of axis <axis> which
                      contains the extracted features.

    References:
        "A large set of audio features for sound description (similarity and classification)
        in the CUIDADO project", Peeters G., 2004.
        URL: http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
        (accessed 11/05/2023)

    NOTE: The peakFinder() function returns nans if less than <numHarmonics> peaks are found. These will
          propagated to the 'pFreqs', 'pAmps' arrays of the temporal_globapy.py features, that populate the
         'globalFeatures' array.
    """

    # Preprocess signals
    tAxis         = axis
    sAxis         = tAxis + 1 # Axis to contain the spectral amplitudes after the FFT
    padSize       = pre.getPadSize(signal.shape[tAxis], frameSize, hopSize)
    signal        = pre.pad(signal, padValue, padSize, padType, tAxis)
    chunks        = pre.chunk(signal, frameSize, hopSize, tAxis)
    chunks        = pre.detrend(chunks, axis = sAxis)

    # Compute FFT, power and octave spectra
    freqs, amps   = pre.spectra.fourier(chunks, fs, sAxis)
    poweramps     = pre.spectra.psd(amps, fs, sAxis)
    _, octaveamps = pre.spectra.octave(freqs, poweramps, octaveDesignator, sAxis)
    octaveampsDB  = pre.powerTodb(octaveamps, octaveamps.max(axis = sAxis, keepdims = True)) 
    powampsDB     = pre.powerTodb(poweramps,  poweramps.max( axis = sAxis, keepdims = True)) 

    # Compute spectrogram in Bark scale
    barkAmps      = pre.scales.spectrogram(poweramps, fs, numFilters, 'bark', sAxis)
    barkScales    = np.arange(numFilters)

    # Evaluate RMS and energy amps
    rmsEnvelope   = pre.envelopes.energy(chunks, envFrameSize, envHopSize, sAxis)
    ampEnvelope   = pre.envelopes.amplitude(chunks, sAxis)

    # Extract harmonic freqs/amps
    pFreqs, pAmps = pre.peakFinder(freqs, powampsDB, harmonics, sAxis)
    hFreqs, hAmps = pre.harmonicModel(freqs, powampsDB, pFreqs, harmonics, sAxis)

    # Compute loudness
    loudness      = perceptual.loudness(freqs, poweramps, fs, 'bark', numFilters, sAxis)

    # Evaluate all (time-frame local) features along the spectral axis
    localFeatures = np.concatenate([
        harmonic.features(freqs, powampsDB, hFreqs, hAmps, pFreqs, pAmps, fs, harmonics, sAxis),
        spectral.features(freqs, np.abs(amps), fs, numMFCC, tAxis, sAxis),
        spectral.featuresSmall(barkScales, barkAmps, tAxis, sAxis),
        temporal.features(ampEnvelope, numLags, axis = sAxis),
        temporal.features(rmsEnvelope, numLags, axis = sAxis),
        temporal.features(chunks,      numLags, axis = sAxis),
        spectral.featuresSmall(hFreqs, hAmps, tAxis, sAxis),
        perceptual.features(loudness, tAxis, sAxis),
        various.tonality(poweramps, sAxis),
        ], axis = sAxis)

    # Get weighted-average of the above features over time. Weights are the instantaneous
    # (per time-frame) loudness values
    weights          = np.nanmean(loudness, axis = sAxis, keepdims = True)
    featuresWeighted = np.nanmean(localFeatures * weights, axis = tAxis) / weights.sum(axis = tAxis)

    # Evaluate global (time-frame independent) features
    globalFeatures = np.concatenate([
        temporal_global.features(freqs, poweramps.mean(axis = tAxis), harmonics, tAxis),
        octaveampsDB.mean(axis = tAxis),
        time.features(signal, tAxis)
        ], axis = tAxis)

    return np.concatenate([featuresWeighted, globalFeatures], axis = tAxis)