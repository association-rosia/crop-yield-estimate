import sys
import os
import pandas as pd

sys.path.append(os.curdir)
from src.constants import get_constants

from src.features.great.models.great import GReaT

cst = get_constants()

path_model = os.path.join(cst.path_models, 'GPT2Impute')
great = GReaT.load_from_dir(path_model)

# file_data_train = os.path.join(cst.path_interim_data, 'TrainToImpute.csv')
# X_train = pd.read_csv(file_data_train)
# X_train, y_train = X_train.drop(columns=cst.target_column), X_train[cst.target_column]
# X_train_imputed = great.impute(X_train, max_length=1024)  # , device='mps')
# save_path = os.path.join(cst.path_interim_data, 'TrainImputed.csv')
# X_train_imputed = pd.read_csv(save_path)
# X_train_imputed = pd.concat([X_train_imputed, y_train], axis='columns')
# X_train_imputed.to_csv(save_path, index=False)

# file_data_test = os.path.join(cst.path_interim_data, 'TestToImpute.csv')
# X_test = pd.read_csv(file_data_test)
# X_test_imputed = great.impute(X_test, max_length=1024)  # , device='mps')
# save_path = os.path.join(cst.path_interim_data, 'TestImputed.csv')
# X_test_imputed.to_csv(save_path, index=False)

file_data_gen = os.path.join(cst.path_generated_data, 'TrainGenerated-6038.csv')
X_gen = pd.read_csv(file_data_gen)
X_gen, y_gen = X_gen.drop(columns=cst.target_column), X_gen[cst.target_column]
X_gen_imputed = great.impute(X_gen, max_length=1024)
save_path = os.path.join(cst.path_generated_data, 'TrainGeneratedImputed-6038.csv')
X_gen_imputed = pd.concat([X_gen_imputed, y_gen], axis='columns')
X_gen_imputed.to_csv(save_path, index=False)
