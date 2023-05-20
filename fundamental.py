import numpy as np

def qint(ym1, y0, yp1):
    #QINT   Quadratic interpolation of 3 uniformly spaced samples
    #
    #               [p,y,a] = qint(ym1,y0,yp1)
    #
    #       returns extremum-location p, height y, and half-curvature a
    #       of a parabolic fit through three points.
    #       The parabola is given by y(x) = a*(x-p)^2+b,
    #       where y(-1)=ym1, y(0)=y0, y(1)=yp1.
    
    p = (yp1 - ym1) / (2 * (2 * y0 - yp1 - ym1))
    y = y0 - 0.25 * (ym1 - yp1) * p
    a = 0.5 * (ym1 - 2 * y0 + yp1)
    return p, y, a


def maxr(a):
    """ Finds interpolated maximizer and max value of an array a """

    m = a.shape[0]
    x = np.argmax(a)
    y = a[x]

    if x > 1 and x < m - 1: # max is not onthe edges
        xdelta, yi, hc = qint(a[x-1], y, a[x+1])
        xi = x + xdelta
    else:
        xi = x    # vector of maximizer locations, one per col of a
        yi = y    # vector of maximum values, one per column of a
        hc = 0

    return xi, yi, hc


def findpeaks(data, npeaks, minwidth, maxwidth, minpeak):
    """ Finds up to <npeaks> interpolated peaks in the data. 
        %       A peak is rejected if its width is
        %         less than minwidth samples wide(default=1), or
        %         less than minpeak in magnitude (default=min(data)).
        %       Quadratic interpolation is used for peak interpolation.
        %       Left-over data with peaks removed is returned in resid.
        %       Peaks are returned in order of decreasing amplitude.
            
    """

    peakamps = np.zeros(npeaks)
    peaklocs = np.zeros(npeaks)
    peakwidths = np.zeros(npeaks)

    nrej, ipeak = 0, 0

    while ipeak < npeaks:
        ploc, pamp, pcurv = maxr(data)

        if pamp == minpeak:
            print('findpeaks:min amp reached')
            break

        plocq = np.round(ploc)
        ulim = int(min(len(data), plocq+1))
        camp = pamp


        # Follow peak down to determine its width
        drange = np.max(data) - minpeak
        tol = drange * 0.01
        dmin = camp

        while ulim < len(data) and data[ulim] <= dmin + tol:
            camp = data[ulim]
            ulim += 1
            if camp < dmin:
                dmin = camp

        ulim -= 1
        lamp = camp

        llim = int(max(1, plocq-1))
        camp = pamp
        dmin = camp
        

        while llim > 1 and data[llim] <= dmin + tol:
            camp = data[llim]
            llim -= 1
            if camp < dmin:
                dmin = camp

        llim += 1
        uamp = camp

        # Remove the peak
        data[llim:ulim] = min(lamp, uamp) * np.ones(ulim - llim)

        # Reject peaks which are too narrow (indicated by zero loc and amp)
        pwid = ulim - llim + 1

        if pwid >= minwidth:
            
            peaklocs[ipeak] = ploc
            peakamps[ipeak] = pamp
            peakwidths[ipeak] = -1 / pcurv
            ipeak += 1
            nrej = 0
            
        else:
            nrej += 1

            if nrej >= 10:
                print('*** findpeaks: giving up (10 rejected peaks in a row)')
                break

    return peakamps, peaklocs, peakwidths