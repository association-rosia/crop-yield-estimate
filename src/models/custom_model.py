import os
import sys

from pandas import Series, DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from numpy import ndarray

from src.constants import get_constants

cst = get_constants()

class CustomEstimator(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            estimator_cls: BaseEstimator,
            estimator_reg_l: BaseEstimator,
            estimator_reg_m: BaseEstimator,
            estimator_reg_h: BaseEstimator,
            limit_h: int,
            limit_l: int,
    ) -> None:
        
        self.estimator_cls = estimator_cls
        self.estimator_reg_l = estimator_reg_l
        self.estimator_reg_m = estimator_reg_m
        self.estimator_reg_h = estimator_reg_h
        self.limit_h = limit_h
        self.limit_l = limit_l
        
    def create_labels(self, X: DataFrame, y: Series) -> ndarray:
        # 0: Low, 1: Middle, 2: Hight
        yield_by_acre = y / X['Acre']
        y_1 = yield_by_acre < self.limit_h
        y_2 = yield_by_acre < self.limit_l
        
        return y_1.astype(int) + y_2.astype(int)

    def fit(self, X: DataFrame, y: Series):
        y_cls = self.create_labels(X, y)
        self.estimator_cls.fit(X.to_numpy(), y_cls.to_numpy())
        
        X_h = np.where(y_cls == 2, X)
        y_h = np.where(y_cls == 2, y)
        self.estimator_reg_h.fit(X_h, y_h)
        
        X_m = np.where(y_cls == 1, X)
        y_m = np.where(y_cls == 1, y)
        self.estimator_reg_m.fit(X_m, y_m)
        
        X_l = np.where(y_cls == 0, X)
        y_l = np.where(y_cls == 0, y)
        self.estimator_reg_l.fit(X_l, y_l)
    
    def predict(self, X: ndarray) -> ndarray:
        y_cls = self.estimator_cls.predict(X)
        y_pred = np.zeros(y_cls.shape)
        
        X_h = np.where(y_cls == 2, X)
        y_pred_h = self.estimator_reg_h.predict(X_h)
        y_pred = np.where(y_cls != 2, y_pred, y_pred_h)
        
        X_m = np.where(y_cls == 2, X)
        y_pred_m = self.estimator_reg_m.predict(X_m)
        y_pred = np.where(y_cls != 1, y_pred, y_pred_m)
        
        X_l = np.where(y_cls == 2, X)
        y_pred_l = self.estimator_reg_l.predict(X_l)
        y_pred = np.where(y_cls != 0, y_pred, y_pred_l)
        
        return y_pred
        
        