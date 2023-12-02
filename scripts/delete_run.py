import wandb
from tqdm import tqdm

api = wandb.Api(timeout=60)
runs = api.runs('association-rosia/crop-yield-estimate', per_page=500)

while True:
    try:
        for run in tqdm(runs):
            run.delete()
    except Exception as e:
        print(e)
        api = wandb.Api(timeout=60)
        runs = api.runs('association-rosia/crop-yield-estimate', per_page=500)
