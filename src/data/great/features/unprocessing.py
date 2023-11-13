import os
import pandas as pd
from datetime import datetime
from src.constants import get_constants
import math

cst = get_constants()


def one_hot_decoding_row(row, col, ohe_cols):
    value = []

    for ohe_col in ohe_cols:
        if row[ohe_col]:
            value.append(ohe_col.replace(col, ""))

    return ' '.join(value)


def one_hot_decoding(df_gen):
    for col in cst.processor['list_cols']:
        ohe_cols = [ohe_col for ohe_col in df_gen.columns if ohe_col.startswith(col)]
        df_gen[col] = df_gen.apply(lambda row: one_hot_decoding_row(row, col, ohe_cols), axis='columns')
        df_gen.drop(columns=ohe_cols, inplace=True)

    return df_gen


def isnan(value):
    try:
        return math.isnan(float(value))
    except Exception as e:
        return False


def is_valid_date(row):
    is_valid = True
    date_format = '%Y-%m-%d'

    for col in cst.processor['date_cols']:
        date = row[col]

        if not isnan(date):
            try:
                date = datetime.strptime(date, date_format)
            except Exception as e:
                is_valid = False
                print(date, e)
                pass

    return is_valid


def remove_wrong_date(df_gen):
    df_gen['isValidDate'] = df_gen.apply(lambda row: is_valid_date(row), axis='columns')
    df_gen = df_gen[df_gen['isValidDate']]
    df_gen.drop(columns='isValidDate', inplace=True)

    return df_gen


if __name__ == '__main__':
    file_name = 'TrainGReaTGPT2-10000.csv'
    file_path = os.path.join(cst.path_interim_data, file_name)
    df_gen = pd.read_csv(file_path)

    df_gen = one_hot_decoding(df_gen)
    df_gen = remove_wrong_date(df_gen)

    new_file_name = file_name.split('-')[0] + f'-{len(df_gen)}'
    save_path = os.path.join(cst.path_processed_data, new_file_name)
    df_gen.to_csv(save_path, index=False)
