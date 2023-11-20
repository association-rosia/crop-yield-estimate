import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from src.constants import get_constants
import math

cst = get_constants()


class CYEGReaTProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, limit_h: int = None, limit_l: int = None) -> None:
        self.limit_h = limit_h
        self.limit_l = limit_l

    def transform(self, generated_file: str = None, max_target_by_acre: float = None):
        generated_path = os.path.join(cst.path_generated_data, generated_file)
        df = pd.read_csv(generated_path)
        df = self.filter(df, max_target_by_acre)
        df = self.one_hot_decoding(df)
        df = self.remove_wrong_date(df)
        df = self.remove_wrong_cat(df)

        return df

    def filter(self, df, max_target_by_acre):
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

    def remove_wrong_date(self, df):
        df['isValid'] = df.apply(lambda row: self.is_valid_date(row), axis='columns')
        df = df[df['isValid']]
        df.drop(columns='isValid', inplace=True)

        return df

    @staticmethod
    def is_valid_cat(row, df_train):
        for col in cst.processor['cat_cols']:
            if 'Year' not in col:
                valid_values = df_train[col].unique().tolist()

                if row[col] not in valid_values:
                    return False

        return True

    def remove_wrong_cat(self, df):
        df_train = pd.read_csv(os.path.join(cst.file_data_train), index_col='ID')
        df['isValid'] = df.apply(lambda row: self.is_valid_cat(row, df_train), axis='columns')
        df = df[df['isValid']]
        df.drop(columns='isValid', inplace=True)

        return df


if __name__ == '__main__':
    great_processor = CYEGReaTProcessor()
    generated_file = 'TrainGenerated-50000.csv'
    df_gen = great_processor.transform(generated_file=generated_file)

    print()
