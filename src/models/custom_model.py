from typing_extensions import Self
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost.sklearn import XGBClassifier, XGBRegressor
import numpy as np
from numpy import ndarray

from src.constants import get_constants

cst = get_constants()

class CustomEstimator(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            estimator_cls: XGBClassifier,
            estimator_reg_l: XGBRegressor,
            estimator_reg_m: XGBRegressor,
            estimator_reg_h: XGBRegressor,
            limit_h: int,
            limit_l: int,
    ) -> None:
        
        self.estimator_cls = estimator_cls
        self.estimator_reg_l = estimator_reg_l
        self.estimator_reg_m = estimator_reg_m
        self.estimator_reg_h = estimator_reg_h
        self.limit_h = limit_h
        self.limit_l = limit_l

    def fit(self, X: ndarray = None, y: ndarray = None) -> Self:
        
        return Self
    
    def predict(self, X: ndarray) -> ndarray:
        y_cls = self.estimator_cls.predict(X)
        y_pred = np.zeros((len(y_cls),))
        
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
        
        