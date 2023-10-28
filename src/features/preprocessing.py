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


class CYEDataPreProcessor(BaseEstimator, TransformerMixin):
    LIST_COLS = ['LandPreparationMethod', 'NursDetFactor', 'TransDetFactor', 'OrgFertilizers',
                 'CropbasalFerts', 'FirstTopDressFert']

    DATE_COLS = ['CropTillageDate', 'RcNursEstDate', 'SeedingSowingTransplanting', 'Harv_date', 'Threshing_date']

    CAT_COLS = ['District', 'Block', 'CropEstMethod', 'TransplantingIrrigationSource',
                'TransplantingIrrigationPowerSource', 'PCropSolidOrgFertAppMethod', 'MineralFertAppMethod',
                'MineralFertAppMethod.1', 'Harv_method', 'Threshing_method', 'Stubble_use'] + \
               [f'{col}Year' for col in DATE_COLS]

    CORR_LIST_COLS = [('Ganaura', 'OrgFertilizersGanaura'), ('BasalUrea', 'CropbasalFertsUrea'),
                      ('1tdUrea', 'FirstTopDressFertUrea')]

    TO_DEL_COLS = ['Harv_methodmachine', 'OrgFertilizersJeevamrit', 'CropbasalFertsMoP', 'FirstTopDressFertNPK',
                   'TransplantingIrrigationSourceWell', 'TransplantingIrrigationPowerSourceSolar', 'Harv_dateYear2022',
                   'Harv_dateYear2023']

    CORR_AREA_COLS = ['CultLand', 'CropCultLand', 'TransIrriCost', 'Ganaura',
                      'CropOrgFYM', 'Harv_hand_rent', 'BasalUrea', '1tdUrea', '2tdUrea']

    def __init__(
            self,
            config: CYEConfigPreProcessor,
    ) -> None:

        self.config = config.get_params()
        self.to_del_cols = []
        self.to_del_corr_cols = []
        # self.to_fill_cols = []
        self.to_fill_values = {}
        self.unique_value_cols = []
        self.out_columns = []
        self.scaler = StandardScaler()
        self.imputer = KNNImputer()

    def preprocess(self, X: DataFrame) -> DataFrame:
        X = X.copy(deep=True)
        X = self.one_hot_list(X)
        X = self.fill_correlated_cols(X)
        X = self.cyclical_date_encoding(X)
        X = self.one_hot_encoding(X)
        # corr_cols = self.get_correlated_cols(X) # only used for analysis
        X = self.delete_correlated_cols(X)

        return X

    def fit(self, X: DataFrame) -> Self:
        X = self.preprocess(X)

        if self.config['scale'] != 'none':
            X = self.scale_area_columns(X)

        nan_columns = X.isnull().sum() / len(X) * 100
        nan_columns_to_delete = nan_columns > self.config['delna_thr']
        self.to_del_cols = nan_columns_to_delete[nan_columns_to_delete].index.tolist()

        if self.config['fillna']:
            # nan_columns_to_fill = (0 < nan_columns) & (nan_columns <= self.config['delna_thr'])
            # self.to_fill_cols = nan_columns_to_fill[nan_columns_to_fill].index.tolist()
            # self.compute_filling_values(X)
            self.imputer.fit(X)

        if self.config['normalisation']:
            self.scaler.fit(X)

        self.get_unique_value_cols(X)

        self.out_columns = X.columns.tolist()

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = self.preprocess(X)

        if self.config['scale'] != 'none':
            X = self.scale_area_columns(X)

        X = self.make_consistent(X)

        if self.config['fillna']:
            # X = self.fill_numerical_columns(X)
            X = pd.DataFrame(self.imputer.transform(X), index=X.index, columns=X.columns)
            X = self.fix_nan_bias(X)

        if self.config['normalisation']:
            X = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)

        X = self.delete_unique_value_cols(X)
        X = self.delete_empty_columns(X)

        return X

    def fit_transform(self, X: DataFrame, y=None, **fit_params) -> DataFrame:
        return self.fit(X, **fit_params).transform(X)

    def one_hot_list(self, X: DataFrame) -> DataFrame:
        for col in self.LIST_COLS:
            split_col = X[col].str.split().explode()
            split_col = pd.get_dummies(split_col, prefix=col, prefix_sep='', dummy_na=True, drop_first=True)
            split_col = split_col.astype(int).groupby(level=0).max()
            split_col.loc[split_col[f'{col}nan'] == 1] = np.nan
            split_col.drop(columns=f'{col}nan', inplace=True)
            X = X.join(split_col)
            X.drop(columns=col, inplace=True)

        return X

    def fill_correlated_cols(self, X: DataFrame) -> DataFrame:
        for col1, col2 in self.CORR_LIST_COLS:
            X.loc[X[col2] == 0, col1] = 0
            X.drop(columns=col2, inplace=True)

        return X

    def cyclical_date_encoding(self, X: DataFrame) -> DataFrame:
        for col in self.DATE_COLS:
            X[col] = pd.to_datetime(X[col])
            X[f'{col}Year'] = X[col].dt.year.astype('string').str[:4]
            X[f'{col}DayOfYear'] = X[col].dt.dayofyear
            X[f'{col}DayOfYearSin'] = np.sin(2 * np.pi * X[f'{col}DayOfYear'] / 365)
            X[f'{col}DayOfYearCos'] = np.cos(2 * np.pi * X[f'{col}DayOfYear'] / 365)
            X.drop(columns=[col, f'{col}DayOfYear'], inplace=True)

        return X

    def one_hot_encoding(self, X: DataFrame) -> DataFrame:
        for col in self.CAT_COLS:
            ohe_col = pd.get_dummies(X[col], prefix=col, prefix_sep='', dummy_na=True, drop_first=True)
            ohe_col.rename(columns={f'{col}<NA>': f'{col}nan'}, inplace=True)
            ohe_col.loc[ohe_col[f'{col}nan'] == 1] = np.nan
            ohe_col.drop(columns=f'{col}nan', inplace=True)
            X = X.join(ohe_col)
            X.drop(columns=col, inplace=True)

        return X

    # def get_correlated_cols(self, X: DataFrame) -> DataFrame:
    #     corr_mat = X.corr()
    #     corr_cols = []
    #
    #     for col in X.columns:
    #         col_indexes = corr_mat.index[(corr_mat[col] > 0.9) | (corr_mat[col] < -0.9)].tolist()
    #
    #         for col_index in col_indexes:
    #             if col != col_index:  # and col not in self.CORR_COLS and col_index not in self.CORR_COLS:
    #                 corr_cols.append(sorted((col, col_index)))
    #
    #     corr_cols = sorted(corr_cols)
    #
    #     return corr_cols

    def delete_correlated_cols(self, X: DataFrame) -> DataFrame:
        to_del_cols = [col for col in X.columns if col in self.TO_DEL_COLS]
        X.drop(columns=to_del_cols, inplace=True)

        return X

    def delete_empty_columns(self, X: DataFrame) -> DataFrame:
        X.drop(columns=self.to_del_cols, inplace=True)

        return X

    # def fill_numerical_columns(self, X: DataFrame) -> DataFrame:
    #     for col in self.to_fill_cols:
    #         # All columns are not in Test dataset
    #         if col in X.columns:
    #             X[col] = X[col].fillna(self.to_fill_values[col])
    #
    #     return X

    # def compute_filling_values(self, X: DataFrame):
    #     for col in self.to_fill_cols:
    #         if self.config['fill_mode'] == 'mean':
    #             value = X[col].mean()
    #         elif self.config['fill_mode'] == 'median':
    #             value = X[col].median()
    #         else:
    #             raise NotImplementedError('Unknown filling mode')
    #
    #         self.to_fill_values[col] = value

    def scale_area_columns(self, X: DataFrame) -> DataFrame:
        X.loc[:, self.CORR_AREA_COLS] = X[self.CORR_AREA_COLS].divide(X[self.config['scale']], axis='index')
        X.drop(columns=self.config['scale'], inplace=True)

        return X

    def fix_nan_bias(self, X: DataFrame) -> DataFrame:
        for col in X.columns:
            is_in = sum([1 if cl_col in col else 0 for cl_col in self.CAT_COLS + self.LIST_COLS])

            if is_in:
                X[col] = X[col].astype(int)
                X[col] = np.where(X[col] > 0.5, 1, 0)

        return X

    def get_unique_value_cols(self, X: DataFrame):
        for col in X.columns:
            unique_values = X[col].unique().tolist()
            num_unique_values = len(unique_values)
            substract_value = 1 if sum(np.isnan(unique_values)) else 0

            if num_unique_values - substract_value == 1:
                self.unique_value_cols.append(col)

    def delete_unique_value_cols(self, X: DataFrame) -> DataFrame:
        X.drop(columns=self.unique_value_cols, inplace=True)

        return X

    def make_consistent(self, X: DataFrame) -> DataFrame:
        """
        Allows creating the columns that were not created during One Hot Encoding and initializes them to 0.
        Allows deleting the columns created during One Hot Encoding that did not exist during the fit.
        """
        # missing_columns = [col for col in self.out_columns if col not in X.columns]
        # extra_columns = [col for col in X.columns if col not in self.out_columns]
        #
        # for col in missing_columns:
        #     X[col] = 0
        #
        # X.drop(columns=extra_columns, inplace=True)
        X = X[self.out_columns]

        return X


class CYETargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config: CYEConfigTransformer) -> None:
        super().__init__()

        self.config = config.__dict__
        self.scaler = None

    def fit(self, X: DataFrame, y: Series = None) -> Self:
        if self.config['scale'] != 'none':
            self.scaler = X[self.config['scale']].copy(deep=True)

        return self

    def transform(self, y: Series) -> Series:
        if self.config['scale'] != 'none':
            y = y / self.scaler

        return y

    def fit_transform(self, X: DataFrame, y: Series) -> Series:

        return self.fit(X, y).transform(y)

    def inverse_transform(self, y: Series) -> Series:
        if self.config['scale'] != 'none':
            y = y * self.scaler

        return y


if __name__ == '__main__':
    from src.constants import get_constants

    cst = get_constants()
    
    config = CYEConfigPreProcessor(fillna=True)
    processor = CYEDataPreProcessor(config=config)

    # config = CYEConfigTransformer(scale=scale)
    # transformer = CYETargetTransformer(config=config)
    df_train = pd.read_csv(cst.file_data_train, index_col='ID')

    X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
    # y_train = transformer.fit_transform(X_train, y_train)
    X_train = processor.fit_transform(X_train)
    # y_train = transformer.inverse_transform(y_train)

    # Test data
    X_test = pd.read_csv(cst.file_data_test, index_col='ID')
    # y_test = transformer.fit(X_test)
    X_test = processor.transform(X_test)

    print()
