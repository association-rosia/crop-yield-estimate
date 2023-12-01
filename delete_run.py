import wandb
from tqdm import tqdm
from requests.exceptions import ReadTimeout
api = wandb.Api()
runs = api.runs('association-rosia/crop-yield-estimate')

while runs:
    try:
        runs = api.runs('association-rosia/crop-yield-estimate')
        for run in tqdm(runs):
            run.delete()
    except ReadTimeout as e :
        api = wandb.Api()
        runs = api.runs('association-rosia/crop-yield-estimate')
