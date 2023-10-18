import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing_extensions import Self
import joblib

sys.path.append(os.curdir)

from src.constants import get_constants

cst = get_constants()

from src.features.config import CYEConfigPreProcessor


class CYEDataPreProcessor(BaseEstimator, TransformerMixin):
    LIST_COLS = ['LandPreparationMethod', 'NursDetFactor', 'TransDetFactor', 'OrgFertilizers',
                 'CropbasalFerts', 'FirstTopDressFert']

    # num_cols = ['CultLand', 'CropCultLand', 'CropTillageDepth', 'SeedlingsPerPit',
    #             'TransplantingIrrigationHours', 'TransIrriCost', 'StandingWater', 'Ganaura', 'CropOrgFYM',
    #             'NoFertilizerAppln', 'BasalDAP', 'BasalUrea', '1tdUrea', '1appDaysUrea', '2tdUrea',
    #             '2appDaysUrea', 'Harv_hand_rent', 'Residue_length', 'Residue_perc', 'Acre', 'Yield']

    DATE_COLS = ['CropTillageDate', 'RcNursEstDate', 'SeedingSowingTransplanting', 'Harv_date', 'Threshing_date']

    CAT_COLS = ['District', 'Block', 'CropEstMethod', 'TransplantingIrrigationSource',
                'TransplantingIrrigationPowerSource', 'PCropSolidOrgFertAppMethod', 'MineralFertAppMethod',
                'MineralFertAppMethod.1', 'Harv_method', 'Threshing_method', 'Stubble_use'] + [f'{col}Year' for col in
                                                                                               DATE_COLS]

    CORR_LIST_COLS = [('CropOrgFYM', 'OrgFertilizersFYM'), ('Ganaura', 'OrgFertilizersGanaura'),
                      ('BasalDAP', 'CropbasalFertsDAP'), ('BasalUrea', 'CropbasalFertsUrea'),
                      ('1tdUrea', 'FirstTopDressFertUrea')]

    def __init__(
            self,
            config: CYEConfigPreProcessor,
    ) -> None:

        self.config = config.__dict__

        self.to_del_cols = []
        self.to_fill_cols = []
        self.to_fill_values = {}
        self.unique_value_cols = []
        self.out_columns = []
        self.scaler = StandardScaler()

    def one_hot_list(self, X: DataFrame) -> DataFrame:
        for col in self.LIST_COLS:
            split_col = X[col].str.split().explode()
            split_col = pd.get_dummies(split_col, dummy_na=True, prefix=col, prefix_sep='')
            split_col = split_col.astype(int).groupby(level=0).max()
            split_col.loc[split_col[f'{col}nan'] == 1] = np.nan
            split_col.drop(columns=f'{col}nan', inplace=True)
            X = X.join(split_col)
            X.drop(columns=col, inplace=True)

        return X

    def fill_correlated_list(self, X: DataFrame) -> DataFrame:
        for col1, col2 in self.CORR_LIST_COLS:
            X.loc[X[col2] == 0, col1] = 0
            X.drop(columns=col2, inplace=True)

        return X

    def cyclical_date_encoding(self, X: DataFrame) -> DataFrame:
        for col in self.DATE_COLS:
            X[col] = pd.to_datetime(X[col])
            X[f'{col}Year'] = X[col].dt.year.astype('string')
            X[f'{col}DayOfYear'] = X[col].dt.dayofyear
            X[f'{col}DayOfYearSin'] = np.sin(2 * np.pi * X[f'{col}DayOfYear'] / 365)
            X[f'{col}DayOfYearCos'] = np.cos(2 * np.pi * X[f'{col}DayOfYear'] / 365)
            X.drop(columns=[col, f'{col}DayOfYear'], inplace=True)

        return X

    def one_hot_encoding(self, X: DataFrame) -> DataFrame:
        for col in self.CAT_COLS:
            X[col] = X[col].fillna('Unknown')
            ohe_col = pd.get_dummies(X[col], prefix=col, prefix_sep='')
            ohe_col = ohe_col.astype(int).groupby(level=0).max()
            X = X.join(ohe_col)
            X.drop(columns=col, inplace=True)

        return X

    def delete_empty_columns(self, X: DataFrame) -> DataFrame:
        X.drop(columns=self.to_del_cols, inplace=True)

        return X

    def fill_numerical_columns(self, X: DataFrame) -> DataFrame:
        for col in self.to_fill_cols:
            # All columns are not in Test dataset
            if col in X.columns:
                X[col] = X[col].fillna(self.to_fill_values[col])

        return X

    def compute_filling_values(self, X: DataFrame):
        for col in self.to_fill_cols:
            if self.config['fill_mode'] == 'mean':
                value = X[col].mean()
            elif self.config['fill_mode'] == 'median':
                value = X[col].median()
            else:
                raise NotImplementedError('Unknown filling mode')

            self.to_fill_values[col] = value

    def preprocess(self, X: DataFrame) -> DataFrame:
        X.set_index('ID', inplace=True)
        X = self.one_hot_list(X)
        X = self.fill_correlated_list(X)
        X = self.cyclical_date_encoding(X)

        if self.config['fillna']:
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

    def make_consistent(self, X: DataFrame) -> DataFrame:
        """
        Allows creating the columns that were not created during One Hot Encoding and initializes them to 0.
        Allows deleting the columns created during One Hot Encoding that did not exist during the fit.
        """
        missing_columns = [col for col in self.out_columns if col not in X.columns]
        extra_columns = [col for col in X.columns if col not in self.out_columns]

        for col in missing_columns:
            X[col] = 0

        X.drop(columns=extra_columns, inplace=True)
        X = X[self.out_columns]

        return X

    def fit(self, X: DataFrame) -> Self:
        nan_columns = X.isnull().sum() / len(X) * 100
        nan_columns_to_delete = nan_columns > self.config['delna_thr']
        self.to_del_cols = nan_columns_to_delete[nan_columns_to_delete].index.tolist()

        if self.config['fillna']:
            nan_columns_to_fill = (0 < nan_columns) & (nan_columns <= self.config['delna_thr'])
            self.to_fill_cols = nan_columns_to_fill[nan_columns_to_fill].index.tolist()

            self.compute_filling_values(X)
            self.get_unique_value_cols(X)

        if self.config['normalisation']:
            self.scaler.fit(X)

        self.out_columns = X.columns.tolist()

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if self.config['fillna']:
            X = self.fill_numerical_columns(X)

        X = self.delete_unique_value_cols(X)
        X = self.delete_empty_columns(X)
        X = self.make_consistent(X)

        if self.config['normalisation']:
            X = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)

        return X

    def save(self, save_folder: str) -> bool:
        dict_to_save = {
            'config': self.config,
            'to_del_cols': self.to_del_cols,
            'to_fill_cols': self.to_fill_cols,
            'to_fill_values': self.to_fill_values,
            'unique_value_cols': self.unique_value_cols,
            'out_columns': self.out_columns,
        }

        os.makedirs(save_folder, exist_ok=True)

        dpp_file_path = os.path.join(save_folder, 'dpp.pkl')
        with open(dpp_file_path, 'w') as f:
            json.dump(dict_to_save, f)

        scaler_file_path = os.path.join(save_folder, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_file_path)

        return dpp_file_path, scaler_file_path

    @classmethod
    def load(cls, dpp_file_path, scaler_file_path, **kwargs: Any) -> Self:
        with open(dpp_file_path, 'r') as f:
            dict_params: dict = json.load(f)

        dict_params.update(kwargs)
        config_preprocessor = CYEConfigPreProcessor(**dict_params['config'])
        self = cls(config_preprocessor)

        self.scaler = joblib.load(scaler_file_path)

        for key, value in dict_params.items():
            if not key == 'config':
                self.__setattr__(key, value)

        return self


if __name__ == '__main__':
    from src.constants import get_constants

    cst = get_constants()

    config = CYEConfigPreProcessor()
    dpp = CYEDataPreProcessor(config=config)
    df_train = pd.read_csv(cst.file_data_train)
    df_train = dpp.preprocess(df_train)

    X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
    X_train = dpp.fit_transform(X_train)

    dpp_config_str = f'dpp_{config.fillna}_{config.delna_thr}_{config.fill_mode}'
    save_folder = os.path.join(cst.path_models, dpp_config_str)
    dpp_file_path, scaler_file_path = dpp.save(save_folder)

    # Test data
    dpp = CYEDataPreProcessor.load(dpp_file_path, scaler_file_path)
    X_test = pd.read_csv(cst.file_data_test)
    X_test = dpp.preprocess(X_test)
    X_test = dpp.transform(X_test)

    print()
