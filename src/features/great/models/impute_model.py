import sys
import os
import pandas as pd

sys.path.append(os.curdir)
from src.constants import get_constants

from src.features.great.models.great import GReaT

cst = get_constants()

path_model = os.path.join(cst.path_models, 'GPT2Impute')
great = GReaT.load_from_dir(path_model)

file_data_train = os.path.join(cst.path_interim_data, 'TrainToImpute.csv')
file_data_test = os.path.join(cst.path_interim_data, 'TestToImpute.csv')
target_column = 'Yield'

X_train = pd.read_csv(file_data_train)
X_train_imputed = great.impute(X_train, max_length=1024, device='mps')
save_path = os.path.join(cst.path_interim_data, 'TrainImputed.csv')
X_train_imputed.to_csv(save_path, index=False)

X_test = pd.read_csv(file_data_test)
X_test_imputed = great.impute(X_test, max_length=1024, device='mps')
save_path = os.path.join(cst.path_interim_data, 'TestImputed.csv')
X_test_imputed.to_csv(save_path, index=False)


