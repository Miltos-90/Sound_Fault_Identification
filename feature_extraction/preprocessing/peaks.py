""" This module implements a peak finding algorithm. """

from .helpers import extract, removeSubharmonics
from typing  import Tuple
import numpy as np


def peakFinder(
    frequencies: np.array, amplitudes: np.array, axis: int, numPeaks: int = 10, minPadFactor: int = 1,
    subHarmonicLimit: float = 0.75, maxRejected: int = 50, minRelativeAmplitude: int = -40
    ) -> Tuple[np.array, np.array]:
    """ Extracts peaks from a spectrum. 

        Inputs:
            frequencies [Hz]: 
                Vector of frequencies for the corresponding FFT amplitudes [Num. frequencies]
            amplitudes [dBFS]: 
                Array of FFT amplitudes. It can have an arbitrary number of dimensions, so long as 
                the input axis <axis> contains <Num. frequencies> elements.
            axis: 
                Axis along which to search for the harmonic amplitudes and frequencies
            numPeaks: 
                Number of peaks to be extracted. A smaller number of peaks will be extracted if
                less are present in the spectra.
            minPadFactor: 
                Minimum zero-padding factor to be used for the peak extraction 
                approximately equal to 5 should be used for the generalized Blackman family of windows
            subHarmonicLimit: Maximum harmonic number below which extracted peaks are rejected
                (used for the removal of the DC term, subharmonics, etc.)
            maxRejected: Num. of consecutive peak rejections before the algorithm 
                stops searching for peaks. It is used in case the number 
                of peaks requested <numPeaks> is higher than the number of 
                peaks that actually exist in the signal above the minimum value <minlevel>
            minRelativeAmplitude:
                Lowest relative partial amplitude a peak should have to be accepted
                -40 [dBFS] should be used with the Hamming window family
                -60 [dBFS] should be with Blackman window family
        
        Outputs:
            peakFreqs: Harmonic frequencies
            peakAmps : Harmonic amplitudes
                Dimensions are exactly the same as the dimensions of the input <amplitudes> matrix, with the
                excpetion of the axis <axis>. The latter will contain <numPeaks> elements instead of
                <Num. frequencies> elements.
    """

    # Compute required constants
    frameSize = 2 * frequencies.shape[0]
    nfft      = 2 ** int( np.ceil( np.log2(frameSize * minPadFactor) ) )
    minAmp    = np.max(amplitudes) + minRelativeAmplitude
    minWidth  = minPadFactor * nfft / frameSize

    # Extract and sort peaks
    amps      = amplitudes.copy() # Make a copy as it will modified in-place in the findPeaks() function
    locs, amps, widths = _findPeaks(amps, numPeaks, minWidth, minAmp, maxRejected, axis)
    ix        = np.argsort(locs, axis)
    locs      = np.take_along_axis(locs, ix, axis)
    amps      = np.take_along_axis(amps, ix, axis)
    widths    = np.take_along_axis(widths, ix, axis)

    peakFreqs, _ = removeSubharmonics(frequencies[locs], subHarmonicLimit, axis)
    peakAmps = extract(x = frequencies, y = amplitudes, xq = peakFreqs, axis = axis)

    # If a peak does not exist, zeros are returned from the harmonic model. Convert to nans
    notFound = peakAmps == 0 
    peakAmps[notFound] = np.nan

    return peakFreqs, peakAmps

    
def _findPeaks(
    x: np.array, npeaks: int, minwidth: int, minpeak: float, 
    maxRejected: int, axis: int) -> Tuple[np.array, np.array, np.array]:
    """ Finds up to <npeaks> interpolated peaks in the matrix <x> along axis <axis>. 
        Inputs:
            x           : Matrix from which peaks will be extracted
            npeaks      : Numebr of peaks to be extracted
            minwidth    : Minimum width of a peak that will be accepted
            minwidth    : Maximum width of a peak that will be accepted
            minpeak     : Minimum amplitude of a peak that will be accepted
            axis        : Axis of x from which the peaks will be extracted
            maxRejected : Maximum number of consecutive peaks to be rejected before the algorithm stops.
                          It is used in case the number of peaks requested <nPeaks> is higher than the number of 
                          peaks that actually exist in the signal above the minimum value <minlevel>
        Outputs:
            peakAmps    : Aplitudes of the accepted peaks
            peakLocs    : Locations (indices) of the accepted peaks
            peakWidths  : Widths of the accepted peaks
        NOTE: If less than <npeaks> peaks exist, a smaller number of peaks will be returned.
    """

    sh          = list(x.shape) # Shape that the output arrays should have
    sh[axis]    = 1
    numDims     = x.ndim
    numSamples  = x.shape[axis]
    stop        = np.zeros(sh, dtype = bool)
    ipeak       = np.zeros(sh, dtype = int)
    numRejected = np.zeros(sh, dtype = int)
    sh[axis]    = npeaks
    peakLocs    = np.zeros(sh, dtype = int)
    peakAmps    = np.zeros(sh)
    peakWidths  = np.zeros(sh)

    while not(np.all(stop)) and not(ipeak.max() == npeaks): 

        ploc, pamp, pcurv = _interpolatedMaxValues(x, axis)
        tol        = 1e-2 * (np.max(x, axis = axis, keepdims = True) - minpeak)
        ulim, camp = _getPeakEnd(ploc, pamp, tol, x, axis)
        llim       = _getPeakStart(ploc, pamp, tol, x, axis)
        x          = _removePeak(camp, ulim, llim, x, axis)
        stop[pamp <= minpeak] = True # Min amplitude reached. stop
        acc, mask  = _acceptPeaks(llim, ulim, minwidth, stop, ipeak, peakLocs, numDims, axis)
        
        # Append to outputs
        peakLocs[mask]   = ploc[acc]
        peakAmps[mask]   = pamp[acc]
        peakWidths[mask] = -1 / pcurv[acc]

        # Increment counter and update exit criteria
        ipeak[acc]        += 1
        numRejected[~acc] += 1
        numRejected[acc]  = 0
        stop[numRejected  >= maxRejected] = True

    return peakLocs, peakAmps, peakWidths


def _quadraticInterpolation(yl: np.array, yc: np.array, yr: np.array
    ) -> Tuple[np.array, np.array, np.array]:
    """ Quadratic interpolation of 3 uniformly-spaced samples.
        Returns extremum-location <p>, height <y>, and half-curvature <a> of a parabolic fit through the
        y-coordinates of three points (these coordinates can be tensors of arbitrary shape).
        The parabola is given by y(x) = a*(x-p)^2+b, where y(-1) = yl, y(0) = yc, y(1) = yr.
    """

    p = (yr - yl) / (2 * (2 * yc - yr - yl))
    y = yc - 0.25 * (yl - yr) * p
    a = 0.5 * (yl - 2 * yc + yr)

    return p, y, a


def _interpolatedMaxValues(x: np.array, axis: int):
    """ Computes interpolated maximum values (peaks), their locations and curvatures using quadratic interpolation.
        Inputs: 
            x        : Tensor of signals (one axis of which represents FFT spectra)
            axis     : Axis along which the peaks will be computed
        Outputs:
            peakLoc  : Locations (indices) of the peaks along the axis <axis>
            peakAmp  : Amplitudes (values) of the peaks along the axis <axis>
            halfCurve: Half curvature (width) of the peaks along the axis <axis>
    """

    n       = x.shape[axis]
    maxLoc  = np.argmax(x, axis = axis, keepdims = True)
    maxAmp  = np.take_along_axis(x, maxLoc, axis)
    edge    = (maxLoc == n - 1) | (maxLoc == 0) # Maxima located on the edges of the array

    # Save separately the peak amplitudes and locations of the <edge> cases
    edgeLoc = maxLoc[edge]
    edgeAmp = maxAmp[edge]

    # Assume that the maxima that appear on the edges appear in the middle of the array for the below computation.
    # This effectively simplifies generating/indexing subarrays for the <edge> cases
    maxLoc[edge] = n // 2

    _, peakAmp, halfCurve = _quadraticInterpolation(
        yl = np.take_along_axis(x, maxLoc - 1, axis),
        yr = np.take_along_axis(x, maxLoc + 1, axis),
        yc = maxAmp
    )

    # Swap the wrong computations (the fake maxima of the <edge> cases) with the real values
    halfCurve[edge] = 0
    maxLoc[edge]    = edgeLoc
    peakAmp[edge]   = edgeAmp

    return maxLoc.astype(int), peakAmp, halfCurve


def _removePeak(value: np.array, upperLimit: np.array, lowerLimit: np.array, x: np.array, axis: int) -> np.array:
    """ Removes a peak from the matrix <x> by masking the relevant values of the matrix with a constant.
        Inputs:
            value     : Values (amplitudes) to replace the elements of the matrix
            upperLimit: Maximum indices along the dimension <axis> that will be replaced
            lowerLimit: Minimum indices along the dimension <axis> that will be replaced
            x         : Matrix for the operation (arbitrary shape)
            axis      : Axis along which the operation will occur
        Outputs:
            x: Input matrix with its peak removed
        Notes:
            val, ulim, llim are vectors with x.shape[axis] elements
    """

    ax   = [a for a in range(x.ndim) if a != axis]
    ind  = np.expand_dims(np.arange(x.shape[axis]), ax)
    mask = ( ind >= lowerLimit ) & (ind < upperLimit)
    np.putmask(x, mask, value)
    
    return x


def _getPeakEnd(peakLoc: np.array, peakAmp: np.array, tol: np.array, x: np.array, axis: int) -> Tuple[np.array, np.array]:
    """ Computes the location (index) of the end of the current peak.
        Inputs:
            peakLoc : Locations of the peaks
            peakAmp : Amplitudes of the peaks
            tol     : Amplitude tolerances
            x       : Matrix for the operation (arbitrary shape)
            axis    : Axis along which the operation will occur
        Outputs:
            upLim   : Input matrix with its peak removed
            cAmp    : Peak center amplitudes
        Notes:
            peakLoc, peakAmp, tol are vectors with x.shape[axis] elements
    """

    n          = x.shape[axis]
    upLim      = np.minimum(n * np.ones_like(peakLoc), peakLoc + 1)
    cAmp, dmin = peakAmp.copy(), peakAmp.copy()
    follow     = (upLim < n-1) & (np.take_along_axis(x, np.minimum(n-1, upLim), axis) <= dmin + tol)

    while np.any(follow):

        cAmp[follow] = np.take_along_axis(x, np.minimum(n-1, upLim), axis)[follow] 
        upLim[follow] += 1
        c       = (cAmp < dmin) & (follow)
        dmin[c] = cAmp[c]
        follow  = (upLim < n-1) & (np.take_along_axis(x, np.minimum(n-1, upLim), axis) <= dmin + tol)

    upLim -= 1

    return upLim, cAmp


def _getPeakStart(peakLoc: np.array, peakAmp: np.array, tol: np.array, x: np.array, axis: int) -> np.array:
    """ Computes the location (index) of the start of the current peak.
        Inputs:
            peakLoc : Locations of the peaks
            peakAmp : Amplitudes of the peaks
            tol     : Amplitude tolerances
            x       : Matrix for the operation (arbitrary shape)
            axis    : Axis along which the operation will occur
        Outputs:
            lowLim  : Input matrix with its peak removed
            cAmp    : Peak center amplitudes
        Notes:
            peakLoc, peakAmp, tol are vectors with x.shape[axis] elements
    """

    lowLim = np.maximum(1, peakLoc-1)
    cAmp   = peakAmp.copy()
    dmin   = peakAmp.copy()
    follow = (lowLim > 1) & (np.take_along_axis(x, np.maximum(1, lowLim), axis) <= dmin + tol)

    while np.any(follow):
        cAmp[follow] = np.take_along_axis(x, np.maximum(1, lowLim), axis)[follow]
        lowLim[follow] -= 1
        c = (cAmp < dmin) & follow
        dmin[c] = cAmp[c]
        follow  = (lowLim > 1) & (np.take_along_axis(x, np.maximum(1, lowLim), axis) <= dmin + tol)

    lowLim += 1

    return lowLim


def _acceptPeaks(
    lowLim: np.array, upLim: np.array, minWidth: int, stop: np.array, 
    ipeak: np.array, peaklocs: np.array, ndim: int, axis: int) -> Tuple[np.array, np.array]:
    """ Computes the location (index) of the start of the current peak.
        Inputs:
            lowLim   : Start indices of the peaks
            upLim    : End indices of the peaks
            minWidth : Minimum width a peak should have to be accepted
            stop     : Vector of indicating the vectors of <axis> that iterations have already stopped
            ipeak    : Vector counter for the number of peaks found so far
            peaklocs : Locations of the center of the peaks
            ndim     : Number of dimensions of the matrix being operated on
            axis     : Axis of the matrix being operated on along which the operation will occur
        Outputs:
            accepted : Vector indicating the dimensions along which the current peaks are accepted
            mask     : Boolean mask indicating on which elements of the output matrix the peak data
                       will be added on.
        Notes:
            lowLim, upLim, stop, ipeak are vectors with numPeak elements
    """

    accepted = (upLim - lowLim + 1 >= minWidth) & (~stop) # Reject narrow peaks
    ax       = [a for a in range(ndim) if a != axis]
    peakRow  = np.expand_dims(np.arange(peaklocs.shape[axis]), ax) == ipeak
    mask     = peakRow & accepted

    return accepted, mask