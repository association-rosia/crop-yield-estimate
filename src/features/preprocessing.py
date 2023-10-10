import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DataPostProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, missing_thr=50, fill_mode='median'):
        self.missing_thr = missing_thr
        self.fill_mode = fill_mode

        self.list_cols = ['LandPreparationMethod', 'NursDetFactor', 'TransDetFactor', 'OrgFertilizers',
                          'CropbasalFerts', 'FirstTopDressFert']

        self.cat_cols = ['District', 'Block', 'CropEstMethod', 'TransplantingIrrigationSource',
                         'TransplantingIrrigationPowerSource', 'PCropSolidOrgFertAppMethod', 'MineralFertAppMethod',
                         'MineralFertAppMethod.1', 'Harv_method', 'Threshing_method', 'Stubble_use']

        self.num_cols = ['CultLand', 'CropCultLand', 'CropTillageDepth', 'SeedlingsPerPit',
                         'TransplantingIrrigationHours', 'TransIrriCost', 'StandingWater', 'Ganaura', 'CropOrgFYM',
                         'NoFertilizerAppln', 'BasalDAP', 'BasalUrea', '1tdUrea', '1appDaysUrea', '2tdUrea',
                         '2appDaysUrea', 'Harv_hand_rent', 'Residue_length', 'Residue_perc', 'Acre', 'Yield']

        self.date_cols = ['CropTillageDate', 'RcNursEstDate', 'SeedingSowingTransplanting', 'Harv_date',
                          'Threshing_date']

        self.corr_list_cols = [('CropOrgFYM', 'OrgFertilizersFYM'), ('Ganaura', 'OrgFertilizersGanaura'),
                               ('BasalDAP', 'CropbasalFertsDAP'), ('BasalUrea', 'CropbasalFertsUrea'),
                               ('1tdUrea', 'FirstTopDressFertUrea')]

        self.to_delete_cols = []
        self.to_fill_values = []
        self.unique_value_cols = []

    def fit_transform(self, X, y=None, **fit_params: dict):
        X = X.set_index('ID')

        for col in self.list_cols:
            split_col = X[col].str.split().explode()
            split_col = pd.get_dummies(split_col, prefix=col, prefix_sep='')
            split_col = split_col.astype(int).groupby(level=0).max()
            X = X.join(split_col)
            X = X.drop(columns=[col])

        for col1, col2 in self.corr_list_cols:
            X.loc[X[col2] == 0, col1] = 0
            X = X.drop(columns=[col2])

        for col in self.date_cols:
            X[col] = pd.to_datetime(X[col])
            X[f'{col}Year'] = X[col].dt.year.astype('string')
            X[f'{col}DayOfYear'] = X[col].dt.dayofyear
            X[f'{col}DayOfYearSin'] = np.sin(2 * np.pi * X[f'{col}DayOfYear'] / 365)
            X[f'{col}DayOfYearCos'] = np.cos(2 * np.pi * X[f'{col}DayOfYear'] / 365)
            self.cat_cols.append(f'{col}Year')
            X = X.drop(columns=[col, f'{col}DayOfYear'])

        for col in self.cat_cols:
            X[col] = X[col].fillna('Unknown')
            ohe_col = pd.get_dummies(X[col], prefix=col, prefix_sep='')
            ohe_col = ohe_col.astype(int).groupby(level=0).max()
            X = X.join(ohe_col)
            X = X.drop(columns=[col])

        missing_column = X.isnull().sum() / len(X) * 100 > self.missing_thr
        to_delete_cols = missing_column[missing_column].index.tolist()
        X = X.drop(columns=to_delete_cols)

        missing_column = X.isnull().sum() / len(X) * 100 > 0
        to_fill_cols = missing_column[missing_column].index.tolist()

        for col in to_fill_cols:
            if self.fill_mode == 'mean':
                value = X[col].mean()
                X[col] = X[col].fillna(value)
            elif self.fill_mode == 'median':
                value = X[col].median()
                X[col] = X[col].fillna(value)
            else:
                raise ValueError('Unknown filling mode')

            self.to_fill_values.append({col: value})

        for col in X.columns:
            num_unique_values = len(X[col].unique())

            if num_unique_values == 1:
                X = X.drop(columns=[col])
                self.unique_value_cols.append(col)

        # TODO : scale data

        return X

    def transform(self, X, y=None):
        return


if __name__ == '__main__':
    data_path = '../../data/raw/Train.csv'
    dpp = DataPostProcessor()

    df = pd.read_csv(data_path)
    df = dpp.fit_transform(df)

    print()
