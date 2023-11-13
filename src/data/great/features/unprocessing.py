import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from src.constants import get_constants
import math

cst = get_constants()


class CYEUnProcessor(BaseEstimator, TransformerMixin):

    def transform_save(self, df):
        df = self.transform(df)
        self.save(df)

    def transform(self, df):
        df = self.one_hot_decoding(df)
        df = self.remove_wrong_date(df)
        df = self.remove_wrong_cat(df)

        return df

    @staticmethod
    def save(df):
        new_file_name = file_name.split('-')[0] + f'-{len(df)}.csv'
        save_path = os.path.join(cst.path_processed_data, new_file_name)
        df.to_csv(save_path, index=False)

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
                    print(col, date, e)
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
    file_name = 'TrainGReaTGPT2-10000.csv'
    file_path = os.path.join(cst.path_interim_data, file_name)
    df_gen = pd.read_csv(file_path)

    unprocessor = CYEUnProcessor()
    unprocessor.transform_save(df_gen)
