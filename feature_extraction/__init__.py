"""
This module implements the preprocessing steps mentioned in Section 2 of:

"A large set of audio features for sound description (similarity and classification)
in the CUIDADO project", Peeters G., 2004.

URL: http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
(accessed 11/05/2023)
"""

from . import preprocessing as pre
from . import temporal, spectral, harmonic