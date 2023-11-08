import sys
import os
import time
from be_great import GReaT

sys.path.append(os.curdir)
from src.constants import get_constants

cst = get_constants()

llm = 'DistilGPT2'
load_path = os.path.join(cst.path_models, llm)
great = GReaT.load_from_dir(load_path)

n_samples = 1000
samples = great.sample(n_samples=n_samples,
                       k=2048,
                       max_length=2000,
                       device='cuda')

now = int(time.time())
save_path = os.path.join(cst.path_interim_data, f'{now}-GReaTSamples{llm}-{n_samples}.csv')
samples.to_csv(save_path, index=False)


