""" Collection of training-related functions """

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Tuple
import config
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def classWeights(targets: np.array) -> dict:
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

    wDict  = classWeights(y)
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


def randomSearchCV(
    X: np.array, y: np.array, groups: np.array,
    numSearch   : int  = config.NUM_RS, 
    numFolds    : int  = config.NUM_FOLDS,
    fixedParams : dict = config.FIXED_PARAMS,
    hyperParams : dict = config.HYPER_SPACE,
    ) -> Tuple[dict, float]:
    """ Performs random hyperparameter search with cross-validation.
        
        Inputs:
            X          : Predictors' array w/ dimensions Num. samples x Num. features
            y          : Targets array w/ dimensions: (Num. samples,)
            groups     : Groups (for CV split stratification) w/ dimensions: (Num. samples,)
            numSearch  : Number of (random) search iterations
            numFolds   : Number of cross-validation folds
            fixedParams: Dictionary containing the fixed parameters used during training
            hyperParams: Dictionary containing arrays of hyperparameters to sample from
                         during training.
        
        Outputs:
            bestParams: Best combination of parameters to train the model
            bestScore : Corresponding average error metric on the validation 
                        set over all folds.
    """

    bestScore, bestParams = np.inf, None

    for _ in range(numSearch):
        
        # Make a random hyperparameter sample
        params = {**fixedParams, **_sample(hyperParams)}

        # Run cross-validation
        scores = _crossValidate(X, y, groups, params, numFolds)

        # Store best score and corresponding hyperparameters
        avgScore = scores.mean()
        if  avgScore < bestScore:
            bestScore  = avgScore
            bestParams = params
    
    return bestParams, bestScore