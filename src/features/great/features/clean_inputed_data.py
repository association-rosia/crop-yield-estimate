import os
import sys

sys.path.append(os.curdir)

import pandas as pd
import numpy as np

import math

from datetime import datetime

from src.constants import get_constants

cst = get_constants()


def clean_df(df_imputed, df_to_impute):
    for col in df_imputed.columns:
        if col in cst.processor['cat_cols']:
            values = [value for value in df_to_impute[col].unique().tolist() if not isnan(value)]
            df_imputed[col] = df_imputed[col].apply(lambda x: x if x in values else np.nan)
        elif col in cst.processor['date_cols']:
            df_imputed[col] = df_imputed[col].apply(lambda x: x if is_valid_date(x) else np.nan)
        else:
            if pd.api.types.is_numeric_dtype(df_imputed[col]):
                min_value = df_to_impute[col].min()
                max_value = df_to_impute[col].max()
                df_imputed[col] = df_imputed[col].apply(lambda x: x if min_value <= x <= max_value else np.nan)
            else:
                df_imputed[col] = df_imputed[col].apply(lambda x: clean_object(x))

    return df_imputed


def clean_object(x):
    if x == 'True':
        x = True
    elif x == 'False':
        x = False
    else:
        x = np.nan

    return x


def main():
    df_train_to_impute = pd.read_csv(os.path.join(cst.path_interim_data, 'TrainToImpute.csv'))
    df_test_to_impute = pd.read_csv(os.path.join(cst.path_interim_data, 'TestToImpute.csv'))
    df_to_impute = pd.concat([df_train_to_impute, df_test_to_impute], axis='rows')

    for csv_file in ['TrainImputed.csv', 'TestImputed.csv', 'TrainGeneratedImputed-6038.csv']:

        if os.path.exists(os.path.join(cst.path_interim_data, csv_file)):
            df_imputed_path = os.path.join(cst.path_interim_data, csv_file)
        else:
            df_imputed_path = os.path.join(cst.path_generated_data, csv_file)

        df_imputed = pd.read_csv(df_imputed_path)
        df_imputed = clean_df(df_imputed, df_to_impute)
        df_imputed.to_csv(df_imputed_path, index=False)


def isnan(value):
    try:
        return math.isnan(float(value))
    except Exception as e:
        return False


def is_valid_date(x):
    is_valid = True
    date_format = '%Y-%m-%d'

    if not isnan(x):
        try:
            date = datetime.strptime(x, date_format)

            if date.year > 2023:
                is_valid = False

        except Exception as e:
            is_valid = False
            pass

    return is_valid


if __name__ == '__main__':
    main()
