from .main import extract, tqdmJoblib

"""
Extractor of a large set of features from audio signals, according to [1].
See the main.extract function for the full set of features, inputs and outputs.

Example usage
--------------

numCores: int   # Number of cores to process the signals in parallel
signals : list  # List of signals (numpy arrays) to be processed

pBar = tqdm(desc = "Extracting features", total = len(signals))

with features.tqdmJoblib(pBar) as progressBar:
    features = joblib.Parallel(n_jobs = numCores)(
        joblib.delayed(extract)(signal,...) for signal in signals
        )

------------

References:
"A large set of audio features for sound description (similarity and classification)
in the CUIDADO project", Peeters G., 2004.
URL: http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf
(accessed 11/05/2023)

"""