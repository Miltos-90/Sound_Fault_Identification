""" Collection of training-related functions """

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def _classWeights(targets: np.array) -> dict:
    """ Computes the class weights from a 1d-array of classes (targets).
        Inputs:
            targets: 1D Array with the targets of a multi-class classification problem
        Outputs:
            weightDict: Dictionary of classNames: classWeights
    """

    yBin     = pd.get_dummies(targets)            # Binary matrix w/ dimensions:  Num.samples x Num.classes
    counts   = np.bincount(yBin.values.argmax(1)) # Num samples per target
    nSamples = targets.shape[0]
    nClasses = yBin.max()
    weights  = nSamples / (counts * nClasses)
    wDict    = { k:v for k, v in zip(yBin.columns, weights) }
    
    return wDict


def _sample(d: dict) -> dict:
    """ Selects a random value from each list field of a dictionary.
        Inputs:
            d: Dict of keys:values (of type list)
        Outputs:
            Dict of of keys: random choice(values)
    """
    return {k: np.random.choice(v) for k, v in d.items()}


def _crossValidate(X: np.array, y: np.array, groups: np.array, params: dict, numFolds: int) -> list:
    """ Performs cross validation.
        Inputs:
            X       : Matrix of predictors w/ dimensions: Num. samples x Num. features
            y       : 1D array containing target labels
            groups  : 1D array containing group labels
            params  : Dictionary containing the training parameters of the LightGBM model
            numFolds: Number of folds for cross-validation
        Outputs:
            scores: Array containing the error metric on the validation set
     """

    wDict  = _classWeights(y)
    scores = np.nan * np.ones(numFolds)
    mapper = lambda d, arr: np.vectorize(d.get)(arr)
    cDict  = {k:i for i, (k, _) in enumerate(wDict.items())}
    skf    = StratifiedKFold(
        n_splits = numFolds, shuffle = True, random_state = params['seed']
    )

    for i, (trainIdx, testIdx) in enumerate(skf.split(X, groups)):

        # Split train/test
        Xtrain, ytrain = X[trainIdx, :], y[trainIdx]
        Xtest,  ytest  = X[testIdx, :],  y[testIdx]

        # Compute class weights
        wtrain, wtest  = mapper(wDict, ytrain), mapper(wDict, ytest)
        ytrain, ytest  = mapper(cDict, ytrain), mapper(cDict, ytest)

        # Make lgb datasets
        trainSet = lgb.Dataset(Xtrain, label = ytrain, weight = wtrain)
        testSet  = lgb.Dataset(Xtest, label = ytest, weight = wtest, reference = trainSet)

        # Train model
        model = lgb.train(params,
            train_set       = trainSet,
            valid_sets      = [testSet], 
            valid_names     = ['val'],
            num_boost_round = params['num_boost_round'],
            callbacks       = [lgb.early_stopping(stopping_rounds = params['early_stopping_rounds'])]
            )

        # Get error metric on the validation set
        scores[i] = model.best_score['val'][params['metric']]
    
    return scores