from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBRegressor

import numpy as np
from numpy import ndarray

from scipy import stats

def get_estimator(estimator_name, seed) -> BaseEstimator:
    if estimator_name == 'XGBRegressor':
        estimator = XGBRegressor(random_state=seed) 
        #, tree_method='hist', device='cuda', sampling_method=gradient_based
    else:
        raise NotImplementedError(f'Estimator {estimator_name} are not implemented.')
    
    return estimator

def get_parameters(estimator_name) -> dict:
    if estimator_name == 'XGBRegressor':
        parameters = {
            'learning_rate': [0.01, 0.1, 0.3, 0.5],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.5, 0.7, 1],
            'colsample_bytree': [0.5, 0.7, 1],
            'colsample_bylevel': [0.5, 0.7, 1],
            'colsample_bynode': [0.5, 0.7, 1],
            'n_estimators': [100, 200, 500],
        }
    else:
        raise NotImplementedError(f'Parameters for {estimator_name} are not implemented.')
        
    return parameters


def compute_stratify(continuous_target: ndarray) -> ndarray:
    target_distribution = stats.norm(loc=continuous_target.mean(), scale=continuous_target.std())
    bounds = target_distribution.cdf([0, 1])
    bins = np.linspace(*bounds, num=10)
    
    return np.digitize(continuous_target, bins)


def get_gridsearch(run_config: dict, target: ndarray):
    estimator = get_estimator(estimator_name=run_config['estimator_name'], seed=run_config['seed'])
    parameters = get_parameters(estimator_name=run_config['estimator_name'])
    kfold = StratifiedKFold(n_splits=run_config['n_splits'], shuffle=True, random_state=run_config['seed'])
    stratify = compute_stratify(target)
    
    # Create Score from MSE Loss
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    return GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring=scorer,
        n_jobs=-1,
        refit=True,
        cv=kfold.split(X=stratify, y=stratify),
        verbose=2
    )