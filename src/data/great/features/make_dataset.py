import pandas as pd
from pandas import DataFrame

import numpy as np
import os

from typing_extensions import Self

from src.constants import get_constants

cst = get_constants()


class GReaTPreProcessor:
    def __init__(self) -> None:
        self.cols_to_replace = []

    def fit(self, X: DataFrame) -> Self:
        X = self.one_hot_list(X, step='fit')

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = self.one_hot_list(X, step='transform')
        X = self.num_to_cat(X)

        return X

    def fit_transform(self, X: DataFrame, y=None, **fit_params) -> DataFrame:
        return self.fit(X).transform(X)

    def one_hot_list(self, X: DataFrame, step: str) -> DataFrame:
        for col in cst.processor['list_cols']:
            split_col = X[col].str.split().explode()
            split_col = pd.get_dummies(split_col, prefix=col, prefix_sep='', dummy_na=True, drop_first=True)
            split_col = split_col.astype(np.uint8).groupby(level=0).max()
            split_col.loc[split_col[f'{col}nan'] == 1] = np.nan
            split_col.drop(columns=f'{col}nan', inplace=True)

            if step == 'fit':
                to_add_cols = [col for col in split_col.columns if col not in cst.processor['to_del_cols']]
                self.cols_to_replace += to_add_cols

            if step == 'transform':
                to_join = [col for col in split_col.columns if col in self.cols_to_replace]
                X = X.join(split_col[to_join])
                X.drop(columns=col, inplace=True)

        return X

    def num_to_cat(self, X: DataFrame) -> DataFrame:
        X[self.cols_to_replace] = X[self.cols_to_replace].replace({0: 'False', 1: 'True'})

        return X


if __name__ == '__main__':
    processor = GReaTPreProcessor()

    df_train = pd.read_csv(cst.file_data_train, index_col='ID')
    X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
    X_train = processor.fit_transform(X_train)

    X_test = pd.read_csv(cst.file_data_test, index_col='ID')
    X_test = processor.transform(X_test)

    df_train = pd.concat([X_train, y_train], axis='columns')
    save_path = os.path.join(cst.path_raw_data, 'TrainGReaT.csv')
    df_train.to_csv(save_path, index=False)

    df_train_test = pd.concat([X_train, X_test], axis='rows')
    save_path = os.path.join(cst.path_raw_data, 'TrainTestGReaT.csv')
    df_train_test.to_csv(save_path, index=False)

    df_train_test_yield = pd.concat([df_train, X_test], axis='rows')
    save_path = os.path.join(cst.path_raw_data, 'TrainTestYieldGReaT.csv')
    df_train_test_yield.to_csv(save_path, index=False)

    print()
