from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from src.features.config import CYEConfigPreProcessor, CYEConfigTransformer
from src.features.preprocessing import CYEDataPreProcessor, CYETargetTransformer
from src.utils import create_labels

import pandas as pd
from pandas import DataFrame, Series

import numpy as np
from numpy import ndarray

from sklearn.model_selection import cross_val_predict
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

from wandb.plot import confusion_matrix

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.constants import get_constants

cst = get_constants()


def init_preprocessor(run_config: dict) -> CYEDataPreProcessor:
    config_preprocessor = CYEConfigPreProcessor(**run_config)
    preprocessor = CYEDataPreProcessor(config_preprocessor)

    return preprocessor


def init_transformer(run_config: dict) -> CYETargetTransformer:
    config_transformer = CYEConfigTransformer(**run_config)
    transformer = CYETargetTransformer(config_transformer)

    return transformer


def init_estimator(run_config: dict) -> RegressorMixin | ClassifierMixin:
    estimator_name = run_config['estimator_name']

    if run_config['task'] in ['regression', 'reg_l', 'reg_m', 'reg_h']:
        estimators = cst.reg_estimators
    elif run_config['task'] == 'classification':
        estimators = cst.cls_estimators

    estimator_config = estimators[estimator_name]['config'](**run_config).get_params()
    estimator = estimators[estimator_name]['estimator'](**estimator_config)

    return estimator


def regression_metrics(y_pred, y_true) -> dict:
    metrics = {
        'reg/rmse': mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False),
        'reg/mae': mean_absolute_error(y_pred=y_pred, y_true=y_true),
        'reg/mape': mean_absolute_percentage_error(y_pred=y_pred, y_true=y_true),
        'reg/r2-score': r2_score(y_pred=y_pred, y_true=y_true)
    }

    return metrics


def classification_metrics(y_pred, y_true) -> dict:
    metrics = {
        'cls/accuracy': accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True),
        'cls/f1-score': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
        'cls/recall': recall_score(y_true=y_true, y_pred=y_pred, average='macro'),
        'cls/precision': precision_score(y_true=y_true, y_pred=y_pred, average='macro'),
        'cls/confusion_matrix': confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=['Low', 'Medium', 'High'],
            title='Confusion matrix',
        )
    }

    return metrics


def init_cross_validator(run_config: dict) -> StratifiedKFold | int:
    if run_config['task'] == 'classification':
        cv = StratifiedKFold(
            n_splits=run_config['cv'],
            shuffle=True,
            random_state=run_config['random_state'],
        )
    else:
        cv = run_config['cv']

    return cv


def init_evaluation_metrics(run_config: dict):
    if run_config['task'] in ['regression', 'reg_l', 'reg_m', 'reg_h']:
        evaluation_metrics = regression_metrics
    elif run_config['task'] == 'classification':
        evaluation_metrics = classification_metrics

    return evaluation_metrics


def get_test_data() -> DataFrame:
    df_test = pd.read_csv(cst.file_data_test, index_col='ID')

    return df_test


def get_train_data(run_config: dict) -> DataFrame:
    df_train = pd.read_csv(cst.file_data_train, index_col='ID')

    if run_config['task'] in ['reg_h', 'reg_m', 'reg_l']:
        labels = create_labels(
            y=df_train[cst.target_column],
            acre=df_train['Acre'],
            limit_h=run_config['limit_h'],
            limit_l=run_config['limit_l'],
        )

        if run_config['task'] == 'reg_l':
            df_train = df_train[labels == 0].copy(deep=True)
        elif run_config['task'] == 'reg_m':
            df_train = df_train[labels == 1].copy(deep=True)
        elif run_config['task'] == 'reg_h':
            df_train = df_train[labels == 2].copy(deep=True)

    return df_train


def apply_smote(X, y) -> (DataFrame, Series):
    k_neighbors = np.unique(y, return_counts=True)[1][2] - 1
    smote = SMOTE(k_neighbors=k_neighbors)
    smoteenn = SMOTEENN(smote=smote)
    X, y = smoteenn.fit_resample(X, y)

    return X, y


def smote_cross_val_predict(estimator, X, y, cv, n_jobs) -> ndarray:
    y_pred = np.zeros(shape=y.shape)

    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        X, y = apply_smote(X, y)
        estimator.fit(X_train, y_train)
        y_pred[val_idx] = estimator.predict(X_val)

    return y_pred


def init_trainer(run_config: dict):
    if run_config['data_aug'] == 'smote':
        trainer = smote_cross_val_predict
    elif run_config['data_aug'] == 'none':
        trainer = cross_val_predict

    return trainer
