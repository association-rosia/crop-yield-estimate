import os
import pandas as pd

from src.constants import get_constants
cst = get_constants()


def one_hot_decoding(row, col, ohe_cols):
    value = []

    for ohe_col in ohe_cols:
        if row[ohe_col]:
            value.append(ohe_col.replace(col, ""))

    return ' '.join(value)


if __name__ == '__main__':
    file_name = '1699613717-GReaTSamplesDistilGPT2-1000.csv'
    file_path = os.path.join(cst.path_interim_data, file_name)
    df_gen = pd.read_csv(file_path)
    df_train = pd.read_csv(cst.file_data_train, index_col='ID')

    for col in cst.processor['list_cols']:
        ohe_cols = [ohe_col for ohe_col in df_gen.columns if ohe_col.startswith(col)]
        df_gen[col] = df_gen.apply(lambda row: one_hot_decoding(row, col, ohe_cols), axis='columns')
        df_gen.drop(columns=ohe_cols, inplace=True)

    save_path = os.path.join(cst.path_processed_data, file_name)
    df_gen.to_csv(save_path)
