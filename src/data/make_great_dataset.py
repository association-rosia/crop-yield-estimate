import pandas as pd
import numpy as np
import os

from src.features.config import CYEConfigPreProcessor
from src.features.preprocessing import CYEDataPreProcessor

from src.constants import get_constants

cst = get_constants()

COLS_TO_REPLACE = ['LandPreparationMethodFourWheelTracRotavator', 'LandPreparationMethodOther',
                   'LandPreparationMethodTractorPlough', 'LandPreparationMethodWetTillagePuddling',
                   'NursDetFactorIrrigWaterAvailability', 'NursDetFactorLabourAvailability',
                   'NursDetFactorPreMonsoonShowers', 'NursDetFactorSeedAvailability',
                   'TransDetFactorIrrigWaterAvailability', 'TransDetFactorLaborAvailability',
                   'TransDetFactorRainArrival', 'TransDetFactorSeedlingAge',
                   'OrgFertilizersGanaura', 'OrgFertilizersGhanajeevamrit', 'OrgFertilizersJeevamrit',
                   'OrgFertilizersPoultryManure', 'OrgFertilizersPranamrit', 'OrgFertilizersVermiCompost',
                   'CropbasalFertsMoP', 'CropbasalFertsNPK', 'CropbasalFertsNPKS', 'CropbasalFertsOther',
                   'CropbasalFertsSSP', 'CropbasalFertsUrea', 'FirstTopDressFertNPK', 'FirstTopDressFertNPKS',
                   'FirstTopDressFertOther', 'FirstTopDressFertSSP', 'FirstTopDressFertUrea']

config = CYEConfigPreProcessor(deloutliers=True)
processor = CYEDataPreProcessor(config=config)

df_train = pd.read_csv(cst.file_data_train, index_col='ID')
X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
X_train = processor.one_hot_list(X_train)

X_train[COLS_TO_REPLACE] = X_train[COLS_TO_REPLACE].replace(1, 'True')
X_train[COLS_TO_REPLACE] = X_train[COLS_TO_REPLACE].replace(0, 'False')
# X_train = X_train.replace(np.nan, 'NaN')

df_train = pd.concat([X_train, y_train], axis='columns')
save_path = os.path.join(cst.path_interim_data, 'Train_GReaT.csv')
df_train.to_csv(save_path, index=False)
