from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame, Series
from typing_extensions import Self
from sklearn.base import RegressorMixin, ClassifierMixin

import os
import sys

sys.path.append(os.curdir)

from src.constants import get_constants

cst = get_constants()


class IterativeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_name: str = 'XGBoost') -> None:
        super().__init__()
        
        self.estimator_name = estimator_name
        self.to_fill_cols = []
        self.estimators = []
        
    def fit(self, X: DataFrame) -> Self:
        self.to_fill_cols = self.get_fill_cols(X)
        X_imputer = X.copy(deep=True)
        
        for i in range(len(self.to_fill_cols)):
            estimator = self.fit_estimator(X_imputer, i)
            self.estimators.append(estimator)
            X_imputer[self.to_fill_cols[i]] = self.impute_column(X_imputer, i)

        return Self
        
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        for i in range(len(self.to_fill_cols)):
            X[self.to_fill_cols[i]] = self.impute_column(X, i)
            
        return X
    
    def fit_transform(self, X: DataFrame, y = None, **fit_params) -> DataFrame:
        return self.fit(X).transform(X, y)
        
    def get_fill_cols(self, X: DataFrame) -> list:
        dict_columns =  {
            col: X[col].isna().value_counts(normalize=True)[True] 
            for col in X.columns
            if X[col].isna().any() and not X[col].isna().all()
        }
        
        sorted_columns = sorted(dict_columns, key=lambda col: dict_columns[col])
        
        return sorted_columns
    
    def get_estimator(self, y_name: str) -> RegressorMixin | ClassifierMixin:
        if y_name in cst.processor['cat_cols'] or y_name in cst.processor['date_cols']:
            dict_estimator = cst.cls_estimators[self.estimator_name]
        else:
            dict_estimator = cst.reg_estimators[self.estimator_name]
        
        estimator = dict_estimator['estimator']()
        
        return estimator
    
    def impute_column(self, X: DataFrame, idx: int) -> Series:
        y_imputer = X[self.to_fill_cols[idx]].copy(deep=True)
        y_index = y_imputer.isna()
        
        X_imputer = X.drop(columns=self.to_fill_cols[idx:])
        X_nan = X_imputer[y_index]
        
        if len(X_nan) > 0:
            y_imputer.loc[y_index] = self.estimators[idx].predict(X_nan.to_numpy())
        
        return y_imputer
    
    def fit_estimator(self, X: DataFrame, idx: int) -> RegressorMixin | ClassifierMixin:
        y_imputer = X[self.to_fill_cols[idx]]
        y_index = y_imputer.notna()
        y_train = y_imputer[y_index]
        
        X_imputer = X.drop(columns=self.to_fill_cols[idx:])
        X_train = X_imputer[y_index]
        
        estimator = self.get_estimator(y_name=self.to_fill_cols[idx])
        estimator.fit(
            X_train.to_numpy(),
            y_train.to_numpy()
        )
        
        return estimator