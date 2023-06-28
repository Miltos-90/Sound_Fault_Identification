""" Collection of training-related functions """

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def classWeights(targets: np.array) -> dict:
    """ Computes the class weights from a 1d-array of classes (targets).
        Inputs:
            targets: 1D Array with the targets of a multi-class classification problem
        Outputs:
            weightDict: Dictionary of classNames: classWeights
    """

    yBin     = pd.get_dummies(targets)  # Binary matrix w/ dimensions:  Num.samples x Num.classes
    counts   = np.bincount(yBin.values.argmax(1)) # Num samples per target
    nSamples = targets.shape[0]
    nClasses = yBin.max()
    weights  = nSamples / (counts * nClasses)
    wDict    = { k:v for k, v in zip(yBin.columns, weights) }
    
    return wDict


def crossValidate(X: np.array, y: np.array, groups: np.array, params: dict, numFolds: int) -> list:
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
    scores = np.empty(numFolds)
    cDict  = {k:i for i, (k, _) in enumerate(wDict.items())}
    skf    = StratifiedKFold(
        n_splits = numFolds, shuffle = True, random_state = params['seed']
    )
    
    for i, (trainIdx, testIdx) in enumerate(skf.split(X, y, groups = groups)):

        Xtrain, ytrain = X[trainIdx, :], y[trainIdx]
        Xtest,  ytest  = X[testIdx, :],  y[testIdx]

        wtrain   = np.vectorize(wDict.get)(ytrain)
        wtest    = np.vectorize(wDict.get)(ytest)

        ytrain   = np.vectorize(cDict.get)(ytrain)
        ytest    = np.vectorize(cDict.get)(ytest)

        trainSet = lgb.Dataset(Xtrain, label = ytrain, weight = wtrain)
        testSet  = lgb.Dataset(Xtest, label = ytest, weight = wtest, reference = trainSet)

        model = lgb.train(params, trainSet,
            valid_sets      = [testSet], 
            valid_names     = ['val'],
            num_boost_round = params['num_boost_round'],
            callbacks       = [
                lgb.early_stopping(
                    stopping_rounds = params['early_stopping_rounds']
                    )
                ]
            )

        scores[i] = model.best_score['val'][params['metric']]

    return scores
