import sys
import os
import pandas as pd
from be_great import GReaT

sys.path.append(os.curdir)
from src.constants import get_constants

cst = get_constants()

data_path = os.path.join(cst.path_interim_data, 'Train_GReaT.csv')
data = pd.read_csv(data_path)

llm = 'distilgpt2'
great = GReaT(llm, epochs=200, experiment_dir=cst.path_models)

great.fit(data)

os.makedirs(cst.path_models, exist_ok=True)
save_path = os.path.join(cst.path_models, llm)
great.save(save_path)
