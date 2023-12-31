import os
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from typing_extensions import Self

sys.path.append(os.curdir)

from src.constants import get_constants

cst = get_constants()

from src.features.config import CYEConfigPreProcessor, CYEConfigTransformer
from src.features.iterativeimputer import IterativeImputer
from src.utils import create_labels


class CYEPreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: CYEConfigPreProcessor) -> None:
        self.config = config
        self.to_del_cols = []
        self.to_del_corr_cols = []
        self.to_fill_values = {}
        self.unique_value_cols = []
        self.out_columns = []
        self.scaler = StandardScaler()
        self.imputer = self.init_imputer(self.config.fillna)

    def preprocess(self, X: DataFrame, y: Series) -> (DataFrame, Series):
        X = X.copy(deep=True)
        X = self.one_hot_list(X)
        X = self.fill_correlated_cols(X)
        X = self.cyclical_date_encoding(X)
        X = self.one_hot_encoding(X)
        X = self.delete_correlated_cols(X)
        X, y = self.delete_outliers(X, y)
        X, y = self.delete_yield_outliers(X, y)

        return X, y

    def fit(self, X: DataFrame, y: Series = None) -> Self:
        X, y = self.preprocess(X, y)
        self.get_to_del_cols(X)
        X = self.scale_area_columns(X)
        self.fit_imputer(X)
        self.get_unique_value_cols(X)
        self.out_columns = X.columns.tolist()

        return self

    def transform(self, X: DataFrame, y: Series = None):
        X, y = self.preprocess(X, y)
        X = self.scale_area_columns(X)
        X = self.make_consistent(X)
        X = self.fill_missing_values(X)
        X = self.delete_unique_value_cols(X)
        X = self.delete_empty_columns(X)

        if y is not None:
            return X, y
        else:
            return X

    def fit_transform(self, X: DataFrame, y: Series = None, **fit_params) -> DataFrame:
        return self.fit(X, y).transform(X, y)

    @staticmethod
    def init_imputer(imputer_name: str):
        if imputer_name == 'KNNImputer':
            imputer = KNNImputer(keep_empty_features=True)
        elif imputer_name == 'IterativeImputer':
            imputer = IterativeImputer()
        elif imputer_name == 'none':
            imputer = None
        else:
            raise ValueError

        return imputer

    @staticmethod
    def one_hot_list(X: DataFrame) -> DataFrame:
        for col in cst.processor['list_cols']:
            split_col = X[col].str.split().explode()
            split_col = pd.get_dummies(split_col, prefix=col, prefix_sep='')  # , dummy_na=True, drop_first=True)
            split_col = split_col.astype(np.uint8).groupby(level=0).max()
            # split_col.loc[split_col[f'{col}nan'] == 1] = np.nan
            # split_col.drop(columns=f'{col}nan', inplace=True)
            X = X.join(split_col)
            X.drop(columns=col, inplace=True)

        return X

    @staticmethod
    def fill_correlated_cols(X: DataFrame) -> DataFrame:
        for col1, col2 in cst.processor['corr_list_cols']:
            if col1 in X.columns and col2 in X.columns:
                X.loc[X[col2] == 0, col1] = 0
                X.drop(columns=col2, inplace=True)

        return X

    def get_to_del_cols(self, X: DataFrame) -> Self:
        nan_columns = X.isnull().sum() / len(X)
        nan_columns_to_delete = nan_columns > self.config.delna_thr
        self.to_del_cols = nan_columns_to_delete[nan_columns_to_delete].index.tolist()

        return self

    def fit_imputer(self, X: DataFrame) -> Self:
        if self.config.fillna != 'none':
            self.imputer.fit(X)

        return self

    @staticmethod
    def cyclical_date_encoding(X: DataFrame) -> DataFrame:
        for col in cst.processor['date_cols']:
            X[col] = pd.to_datetime(X[col])
            X[f'{col}Year'] = X[col].dt.year.astype('string').str[:4]
            X[f'{col}DayOfYear'] = X[col].dt.dayofyear
            X[f'{col}DayOfYearSin'] = np.sin(2 * np.pi * X[f'{col}DayOfYear'] / 365)
            X[f'{col}DayOfYearCos'] = np.cos(2 * np.pi * X[f'{col}DayOfYear'] / 365)
            X.drop(columns=[col, f'{col}DayOfYear'], inplace=True)

        return X

    @staticmethod
    def one_hot_encoding(X: DataFrame) -> DataFrame:
        for col in cst.processor['cat_cols']:
            ohe_col = pd.get_dummies(X[col], prefix=col, prefix_sep='', dummy_na=True, drop_first=True)
            ohe_col = ohe_col.rename(columns={f'{col}<NA>': f'{col}nan'}).astype(np.uint8)

            if f'{col}nan' in ohe_col:
                ohe_col.loc[ohe_col[f'{col}nan'] == 1] = np.nan
                ohe_col.drop(columns=f'{col}nan', inplace=True)

            X = X.join(ohe_col)
            X.drop(columns=col, inplace=True)

        return X

    def delete_outliers(self, X: DataFrame, y: Series) -> (DataFrame, Series):
        if self.config.deloutliers:
            for col in cst.outliers_thr:
                X = X[(X[col] <= cst.outliers_thr[col]) | (X[col].isna())]

        if y is not None:
            y = y[y.index.isin(X.index)]

        return X, y

    def delete_yield_outliers(self, X: DataFrame, y: Series) -> DataFrame:
        if self.config.yieldoutliers_thr and y is not None:
            yield_by_acre = y / X['Acre']
            X = X[yield_by_acre < self.config.yieldoutliers_thr]
            y = y[y.index.isin(X.index)]

        return X, y

    def fill_missing_values(self, X: DataFrame) -> DataFrame:
        if self.config.fillna != 'none' and len(X) > 0:
            X = pd.DataFrame(self.imputer.transform(X), index=X.index, columns=X.columns)
            X = self.fix_nan_bias(X)

        return X

    @staticmethod
    def delete_correlated_cols(X: DataFrame) -> DataFrame:
        to_del_cols = [col for col in X.columns if col in cst.processor['to_del_cols']]
        X.drop(columns=to_del_cols, inplace=True)

        return X

    def delete_empty_columns(self, X: DataFrame) -> DataFrame:
        to_del_cols = [col for col in X.columns if col in self.to_del_cols]
        X.drop(columns=to_del_cols, inplace=True)

        return X

    def scale_area_columns(self, X: DataFrame) -> DataFrame:
        if self.config.scale != 'none':
            X.loc[:, cst.processor['corr_area_cols']] = X[cst.processor['corr_area_cols']].divide(X[self.config.scale],
                                                                                                  axis='index')
            X.drop(columns=self.config.scale, inplace=True)

        return X

    @staticmethod
    def fix_nan_bias(X: DataFrame) -> DataFrame:
        for col in X.columns:
            is_in = any([cl_col in col for cl_col in cst.processor['cat_cols'] + cst.processor['list_cols']])

            if is_in:
                X[col] = np.where(X[col] > 0.5, 1, 0)
                X[col] = X[col].astype(int)

        return X

    def get_unique_value_cols(self, X: DataFrame) -> Self:
        for col in X.columns:
            unique_values = X[col].unique().tolist()
            num_unique_values = len(unique_values)
            substract_value = 1 if sum(np.isnan(unique_values)) else 0

            if num_unique_values - substract_value == 1:
                self.unique_value_cols.append(col)

        return self

    def delete_unique_value_cols(self, X: DataFrame) -> DataFrame:
        X.drop(columns=self.unique_value_cols, inplace=True)

        return X

    def make_consistent(self, X: DataFrame) -> DataFrame:
        missing_cols = list(set(self.out_columns) - set(X.columns))

        for missing_col in missing_cols:
            X[missing_col] = 0

        X = X[self.out_columns]

        return X


class CYETargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config: CYEConfigTransformer) -> None:
        super().__init__()
        self.config = config
        self.scaler = None
        self.acre = None

    def fit(self, X: DataFrame, y: Series = None) -> Self:
        if self.config.scale != 'none':
            self.scaler = X[self.config.scale].copy(deep=True)

        if self.config.task == 'classification':
            self.acre = X['Acre'].copy(deep=True)

        return self

    def transform(self, y: Series) -> Series:
        if self.config.scale != 'none':
            y = y / self.scaler

        if self.config.task == 'classification':
            y = create_labels(
                y=y,
                acre=self.acre,
                limit_h=self.config.limit_h,
                limit_l=self.config.limit_l,
            )

        return y

    def fit_transform(self, X: DataFrame, y: Series = None, **fit_params) -> Series:
        return self.fit(X, y).transform(y)

    def inverse_transform(self, y: Series) -> Series:
        if self.config.scale != 'none':
            y = y * self.scaler

        return y


if __name__ == '__main__':
    config = CYEConfigPreProcessor(fillna='none', deloutliers=True, yieldoutliers_thr=5000)
    processor = CYEPreProcessor(config=config)

    config = CYEConfigTransformer()
    transformer = CYETargetTransformer(config)
    df_train = pd.read_csv(cst.file_data_train, index_col='ID')

    labels = create_labels(
        y=df_train[cst.target_column],
        acre=df_train['Acre'],
        limit_h=5000,
        limit_l=500,
    )

    df_train_l = df_train[labels == 0].copy(deep=True)
    df_train_m = df_train[labels == 1].copy(deep=True)
    df_train_h = df_train[labels == 2].copy(deep=True)

    for df_train in [df_train_l, df_train_m, df_train_h]:
        X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
        # y_train = transformer.fit_transform(X_train, y_train)

        # processor = CYEPreProcessor(config=config)
        X_train, y_train = processor.fit_transform(X_train, y_train)

        print(len(X_train), len(y_train))
        # y_train = transformer.inverse_transform(y_train)
        X_train.index == y_train.index
        # Test data
        X_test = pd.read_csv(cst.file_data_test, index_col='ID')
        X_test = processor.transform(X_test)

    print()
