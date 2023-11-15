import sys
import os
import pandas as pd
from be_great import GReaT

sys.path.append(os.curdir)
from src.constants import get_constants

cst = get_constants()

os.environ['WANDB_PROJECT'] = 'crop-yield-estimate'

data_path = os.path.join(cst.path_raw_data, 'TrainGReaT.csv')
data = pd.read_csv(data_path)

llm = 'GPT2'
model = GReaT(llm=llm.lower(),
              experiment_dir=cst.path_models,
              epochs=50,
              batch_size=128,
              logging_steps=10
              )

model.fit(data)

os.makedirs(cst.path_models, exist_ok=True)
save_path = os.path.join(cst.path_models, llm)
model.save(save_path)
