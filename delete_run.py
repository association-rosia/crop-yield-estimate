import wandb
from tqdm import tqdm

api = wandb.Api(timeout=600)
runs = api.runs('association-rosia/crop-yield-estimate', per_page=1000)

for run in tqdm(runs):
    run.delete()
