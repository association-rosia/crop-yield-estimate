import sys
import os
import pandas as pd
from be_great import GReaT

sys.path.append(os.curdir)
from src.constants import get_constants

cst = get_constants()

llm = 'DistilGPT2'
load_path = os.path.join(cst.path_models, llm)
model = GReaT.load_from_dir(load_path)

data_path = os.path.join(cst.path_raw_data, 'TrainGReaT.csv')
data = pd.read_csv(data_path)

imputed_data = model.impute(data, max_length=2048)

save_path = os.path.join(cst.path_generated_data, f'TrainImputed-{llm}.csv')
imputed_data.to_csv(save_path, index=False)


