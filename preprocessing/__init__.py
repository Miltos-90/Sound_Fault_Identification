"""
This module implements the preprocessing steps mentioned in Section 2 of:

"A large set of audio features for sound description (similarity and classification)
in the CUIDADO project", Peeters G., 2004.

URL: http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
(accessed 11/05/2023)
"""

from .array import chunk, detrend
from . import spectra
from .spectra import todb
from . import envelopes
from . import filters
from .harmonic_model import harmonicModel
from .mel import energy as melEnergy