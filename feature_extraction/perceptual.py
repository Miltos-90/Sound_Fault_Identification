""" This module contains all functions that extract the perceptual features of the sound signals. """

import numpy as np
from typing import Literal
from . import preprocessing as pre


def features(loudness: np.array, timeAxis: int, spectralAxis: int) -> np.array:
    """ Extracts some spectral features in the frequency-domain from an array of signals.
        Inputs:
            loudness    : Array of specific loudness (i.e.e loudness associated with each Bark band).
            timeAxis    : Axis along which the amplitudes are arranged over time (time frame/window-axis).
            spectralAxis: Axis along which the amplitudes are arranged over frequency (frequency-axis).
        Outputs:
            features: Array of features extracted in the time-domain. The output array
                      has the same dimensions as the input signal array, with the exception of
                      axis <scaleAxis>, which contains 27 elements (i.e. features).
    """

    totalLoudness = np.sum(loudness, axis = spectralAxis, keepdims = True)
    relLoudness   = loudness / totalLoudness     # Relative specific loudness
    sharp         = _sharpness(loudness, axis = spectralAxis)
    spread        = _spread(loudness, totalLoudness, axis = spectralAxis)
    totalLoudness = pre.powerTodb(totalLoudness) # Convert from power amplitudes to dB

    # Make output array
    out = np.concatenate([totalLoudness, relLoudness, sharp, spread], axis = spectralAxis)

    return out


def loudness(
    frequencies: np.array, amplitudes: np.array, sampleFrequency: int, 
    scale: Literal["mel", "bark"], numFilters: int, axis: int) -> np.array:
    """ Computes the per-frame perceptual loudness (weighted power) [dB].
    Inputs:
        frequencies     : Corresponding frequency vector for the amplitudes array
        amplitudes      : Power amplitudes array
        sampleFrequency : Sampling rate [Hz]
        scale           : Scale to be used for the loudness (Mel or Bark)
        numFilters      : Number of filters (bins) for the Mel/Bark scale
        axis            : Frequency axis of the amplitudes array
    Outputs:
        loudness: Perceptual loudness per Bark band [power amps]. It's dimensions 
                          match the dimensions of the amplitudes array, wwith the exception
                          of axis <axis> which contains <numFilters> elements.
    References:
        Code adapted from: https://github.com/librosa/librosa/issues/463
    """

    # A-weighting
    weighting = pre.filters.Aweighting(frequencies)  # Weighting matrix [dB]
    weighting = 10 ** (weighting / 10)               # Conversion to apply on the power amplitudes
    weighting = pre.expand(weighting, numDims = amplitudes.ndim, axis = axis)

    # Compute perceptually weighted power spectrogram
    spectrum  = pre.scales.spectrogram(amplitudes * weighting, sampleFrequency,
        numFilters = numFilters, scale = scale, axis = axis)

    return spectrum


def _sharpness(loudness: np.array, axis: int) -> np.array:
    """ Computes acoustic sharpness using the specific loudness of the bark bands.
        Inputs:
            loudness : Array of (specific) loudness of the signals.
            axis     : Axis containing the Bark bands
        Outputs:
            sharp: Array of acoustic sharpness. It's dimensions match the dimensions 
                   of the input loudness array, with axis <axis> missing.
    """

    # Make gain curve
    numBands = loudness.shape[axis]
    bandNum  = np.arange(numBands)
    gain     = np.ones(shape = numBands)
    c        = bandNum > 15
    gain[c]  = 0.066 * np.exp(0.171 * bandNum[c])

    # Fix dimensions
    ndim     = loudness.ndim
    gain     = pre.expand(gain, numDims = ndim, axis = axis)
    bandNum  = pre.expand(np.arange(numBands), numDims = ndim, axis = axis)

    # Compute sharpness
    num   = np.sum(loudness * gain * bandNum, axis = axis, keepdims = True)
    den   = np.sum(loudness, axis = axis, keepdims = True)
    sharp = 0.11 *  num / den

    return sharp


def _spread(specificLoudness: np.array, totalLoudness: np.array, axis: int) -> np.array:
    """ Computes spectral spread, i.e.e the distance between the largest specific loudness
        value to the total loudness.
        Inputs:
            specificLoudness: Loudness [power amps] associated with each Bark band
            totalLoudness   : Total loudness [power amps] of all Bark bands
        Outputs:
            spread: Spectral spread array. It's dimensions match the dimensions of the 
                input arrays, with axis <axis> removed.
    """

    maxLoudness = specificLoudness.max(axis = axis, keepdims = True) 
    return np.square( (totalLoudness - maxLoudness ) / totalLoudness )

