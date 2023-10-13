from typing import Any, cast
from typing_extensions import Self
import json

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import os
import sys

sys.path.append(os.curdir)

from src.features.config import CYEConfigPreProcessor


class CYEDataPreProcessor(BaseEstimator, TransformerMixin):
    list_cols = ['LandPreparationMethod', 'NursDetFactor', 'TransDetFactor', 'OrgFertilizers',
                 'CropbasalFerts', 'FirstTopDressFert']

    # num_cols = ['CultLand', 'CropCultLand', 'CropTillageDepth', 'SeedlingsPerPit',
    #             'TransplantingIrrigationHours', 'TransIrriCost', 'StandingWater', 'Ganaura', 'CropOrgFYM',
    #             'NoFertilizerAppln', 'BasalDAP', 'BasalUrea', '1tdUrea', '1appDaysUrea', '2tdUrea',
    #             '2appDaysUrea', 'Harv_hand_rent', 'Residue_length', 'Residue_perc', 'Acre', 'Yield']

    date_cols = ['CropTillageDate', 'RcNursEstDate', 'SeedingSowingTransplanting', 'Harv_date', 'Threshing_date']
    
    cat_cols = ['District', 'Block', 'CropEstMethod', 'TransplantingIrrigationSource',
                'TransplantingIrrigationPowerSource', 'PCropSolidOrgFertAppMethod', 'MineralFertAppMethod',
                'MineralFertAppMethod.1', 'Harv_method', 'Threshing_method', 'Stubble_use'] + [f'{col}Year' for col in date_cols]

    corr_list_cols = [('CropOrgFYM', 'OrgFertilizersFYM'), ('Ganaura', 'OrgFertilizersGanaura'),
                    ('BasalDAP', 'CropbasalFertsDAP'), ('BasalUrea', 'CropbasalFertsUrea'),
                    ('1tdUrea', 'FirstTopDressFertUrea')]
    def __init__(
        self,
        config: CYEConfigPreProcessor,
    ) -> None:

        self.config = config

        self.to_delete_cols = []
        self.to_fill_cols = []
        self.to_fill_values = {}
        self.unique_value_cols = []

    def one_hot_list(self, X: DataFrame) -> DataFrame:
        for col in self.list_cols:
            split_col = X[col].str.split().explode()
            split_col = pd.get_dummies(split_col, dummy_na=True, prefix=col, prefix_sep='')
            split_col = split_col.astype(int).groupby(level=0).max()
            split_col.loc[split_col[f'{col}nan'] == 1] = np.nan
            split_col.drop(columns=f'{col}nan', inplace=True)
            X = X.join(split_col)
            X.drop(columns=col, inplace=True)

        return X

    def fill_correlated_list(self, X: DataFrame) -> DataFrame:
        for col1, col2 in self.corr_list_cols:
            X.loc[X[col2] == 0, col1] = 0
            X.drop(columns=col2, inplace=True)

        return X

    def cyclical_date_encoding(self, X: DataFrame) -> DataFrame:
        for col in self.date_cols:
            X[col] = pd.to_datetime(X[col])
            X[f'{col}Year'] = X[col].dt.year.astype('string')
            X[f'{col}DayOfYear'] = X[col].dt.dayofyear
            X[f'{col}DayOfYearSin'] = np.sin(2 * np.pi * X[f'{col}DayOfYear'] / 365)
            X[f'{col}DayOfYearCos'] = np.cos(2 * np.pi * X[f'{col}DayOfYear'] / 365)
            X.drop(columns=[col, f'{col}DayOfYear'], inplace=True)

        return X

    def one_hot_encoding(self, X: DataFrame) -> DataFrame:
        for col in self.cat_cols:
            X[col] = X[col].fillna('Unknown')
            ohe_col = pd.get_dummies(X[col], prefix=col, prefix_sep='')
            ohe_col = ohe_col.astype(int).groupby(level=0).max()
            X = X.join(ohe_col)
            X.drop(columns=col, inplace=True)

        return X

    def delete_empty_columns(self, X: DataFrame) -> DataFrame:
        X.drop(columns=self.to_delete_cols, inplace=True)

        return X

    def fill_numerical_columns(self, X: DataFrame) -> DataFrame:
        for col in self.to_fill_cols:
            X[col] = X[col].fillna(self.to_fill_values[col])

        return X

    def compute_filling_values(self, X: DataFrame):
        for col in self.to_fill_cols:
            if self.config.fill_mode == 'mean':
                value = X[col].mean()
            elif self.config.fill_mode == 'median':
                value = X[col].median()
            else:
                raise NotImplementedError('Unknown filling mode')

            self.to_fill_values[col] = value

    def preprocess(self, X: DataFrame) -> DataFrame:
        X.set_index('ID', inplace=True)
        X = self.one_hot_list(X)
        X = self.fill_correlated_list(X)
        X = self.cyclical_date_encoding(X)

        if self.config.fillna:
            X = self.one_hot_encoding(X)

        return X

    def get_unique_value_cols(self, X: DataFrame):
        for col in X.columns:
            num_unique_values = len(X[col].unique())

            if num_unique_values == 1:
                self.unique_value_cols.append(col)

    def delete_unique_value_cols(self, X: DataFrame) -> DataFrame:
        for col in self.unique_value_cols:
            X.drop(columns=col, inplace=True)

        return X

    def fit(self, X: DataFrame):
        nan_columns = X.isnull().sum() / len(X) * 100
        nan_columns_to_delete = nan_columns > self.config.missing_thr
        self.to_delete_cols = nan_columns_to_delete[nan_columns_to_delete].index.tolist()

        if self.config.fillna:
            nan_columns_to_fill = (0 < nan_columns) & (nan_columns <= self.config.missing_thr)
            self.to_fill_cols = nan_columns_to_fill[nan_columns_to_fill].index.tolist()

            self.compute_filling_values(X)
            self.get_unique_value_cols(X)

    def transform(self, X: DataFrame) -> DataFrame:
        X = self.delete_empty_columns(X)
        X = self.delete_unique_value_cols(X)

        if self.config.fillna:
            X = self.fill_numerical_columns(X)

        return X
    
    
    def save_dict(self, path: str) -> bool:
        dict_to_save = {
            'config': self.config.__dict__,
            'to_delete_cols': self.to_delete_cols,
            'to_fill_cols': self.to_fill_cols,
            'to_fill_values': self.to_fill_values,
            'unique_value_cols': self.unique_value_cols,
            'list_cols': self.list_cols,
            'cat_cols': self.cat_cols,
            'date_cols': self.date_cols,
            'corr_list_cols': self.corr_list_cols,
        }
        
        with open(path, 'w') as f:
            json.dump(dict_to_save, f)
            
        return os.path.exists(path)
    
    @classmethod
    def load(cls, path, **kwargs: Any) -> Self:
        with open(path, 'r') as f:
            dict_params: dict = json.load(f)
            
        dict_params.update(kwargs)
        config_preprocessor = CYEConfigPreProcessor(**dict_params['config'])
        self = cls(config_preprocessor)
        
        for key, value in dict_params.items():
            if not key == 'config':
                self.__setattr__(key, value)
        
        return self


if __name__ == '__main__':
    from src.constants import get_constants
    
    cst = get_constants()
    
    config = CYEConfigPreProcessor()
    dpp = CYEDataPreProcessor(config=config)
    data_path = cst.file_data_train
    df = pd.read_csv(data_path)
    df = dpp.preprocess(df)
    dpp.fit(df)
    df = dpp.transform(df)

    # path = os.path.join(cst.path_models, 'test.json')
    # dpp.save_dict(path)
    
    # dpp = CYEDataPreProcessor.load(path)
    df = pd.read_csv(data_path)
    df['District']
    df = dpp.preprocess(df)
    df = dpp.transform(df)
    
    print()