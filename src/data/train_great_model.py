import os
import pandas as pd
from be_great import GReaT

from src.constants import get_constants

cst = get_constants()

data_path = os.path.join(cst.path_interim_data, 'Train_GReaT.csv')
data = pd.read_csv(data_path)

great = GReaT('distilgpt2', epochs=200, experiment_dir=cst.path_models)
trainer = great.fit(data)
