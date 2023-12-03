import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
from datetime import datetime
from src.constants import get_constants
import math
import numpy as np

cst = get_constants()


class GReaTUnprocessor:
    def __init__(self, limit_h: int = None, limit_l: int = None) -> None:
        self.limit_h = limit_h
        self.limit_l = limit_l

    def transform(self, generated_file_path: str = None, split: str = None, max_target_by_acre: float = None):
        df = pd.read_csv(generated_file_path)
        df = self.filter(df, max_target_by_acre)
        df = self.one_hot_decoding(df)
        df = self.remove_wrong_date(df, split)
        df = self.remove_wrong_cat(df, split)

        return df

    def filter(self, df, max_target_by_acre):
        if cst.target_column in df.columns:
            target_by_acre = df[cst.target_column] / df['Acre']
            df.dropna(subset=[cst.target_column], inplace=True)

            if max_target_by_acre:
                df = df[target_by_acre <= max_target_by_acre]

            if self.limit_h and self.limit_l:
                df = df[(target_by_acre > self.limit_h) | (target_by_acre < self.limit_l)]
            elif self.limit_h:
                df = df[target_by_acre > self.limit_h]
            elif self.limit_l:
                df = df[target_by_acre < self.limit_l]

        return df

    @staticmethod
    def one_hot_decoding_row(row, col, ohe_cols):
        value = []

        for ohe_col in ohe_cols:
            if row[ohe_col]:
                value.append(ohe_col.replace(col, ""))

        return ' '.join(value)

    def one_hot_decoding(self, df):
        for col in cst.processor['list_cols']:
            ohe_cols = [ohe_col for ohe_col in df.columns if ohe_col.startswith(col)]
            df[col] = df.apply(lambda row: self.one_hot_decoding_row(row, col, ohe_cols), axis='columns')
            df.drop(columns=ohe_cols, inplace=True)

        return df

    @staticmethod
    def isnan(value):
        try:
            return math.isnan(float(value))
        except Exception as e:
            return False

    def is_valid_date(self, row):
        is_valid = True
        date_format = '%Y-%m-%d'

        for col in cst.processor['date_cols']:
            date = row[col]

            if not self.isnan(date):
                try:
                    date = datetime.strptime(date, date_format)

                    if date.year > 2023:
                        is_valid = False

                except Exception as e:
                    is_valid = False
                    # print(col, date, e)
                    pass

        return is_valid

    def replace_date(self, date):
        is_valid = True
        date_format = '%Y-%m-%d'

        if not self.isnan(date):
            try:
                date = datetime.strptime(date, date_format)

                if date.year > 2023:
                    is_valid = False

            except Exception as e:
                is_valid = False

        if not is_valid:
            date = np.nan

        return date

    def remove_wrong_date(self, df, split):
        if split == 'train':
            df['isValid'] = df.apply(lambda row: self.is_valid_date(row), axis='columns')
            df = df[df['isValid']]
            df.drop(columns='isValid', inplace=True)
        elif split == 'test':
            for col in cst.processor['date_cols']:
                df[col] = df[col].apply(lambda date: self.replace_date(date))
        else:
            raise ValueError

        return df

    @staticmethod
    def is_valid_cat(row, df_train_test):
        for col in cst.processor['cat_cols']:
            if 'Year' not in col:
                valid_values = df_train_test[col].unique().tolist()

                if row[col] not in valid_values:
                    return False

        return True

    @staticmethod
    def replace_cat(cat, series_train_test):
        valid_values = series_train_test.unique().tolist()

        if cat not in valid_values:
            cat = np.nan

        return cat

    def remove_wrong_cat(self, df, split):
        df_train = pd.read_csv(os.path.join(cst.file_data_train), index_col='ID')
        df_test = pd.read_csv(os.path.join(cst.file_data_test), index_col='ID')
        df_train_test = pd.concat([df_train, df_test], axis='rows')

        if split == 'train':
            df['isValid'] = df.apply(lambda row: self.is_valid_cat(row, df_train_test), axis='columns')
            df = df[df['isValid']]
            df.drop(columns='isValid', inplace=True)
        elif split == 'test':
            for col in cst.processor['cat_cols']:
                if 'Year' not in col:
                    df[col] = df[col].apply(lambda cat: self.replace_cat(cat, df_train_test[col]))
        else:
            raise ValueError

        return df


if __name__ == '__main__':
    great_unprocessor = GReaTUnprocessor()
    generated_file_path = os.path.join(cst.path_interim_data, 'TestImputed.csv')
    df_gen = great_unprocessor.transform(generated_file_path=generated_file_path, split='test')

    print()
