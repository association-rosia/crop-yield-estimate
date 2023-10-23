import argparse
import os
import sys
import argparse
from multiprocessing import Process
import yaml

sys.path.append(os.curdir)

import pandas as pd

import wandb

from src.models.utils import init_preprocessor, init_estimator, init_transformer

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

from src.constants import get_constants

cst = get_constants()


def main():
    # Configure & Start run
    run_config = parse_args()
    
    # Load sweep config
    path_sweep = os.path.join(cst.path_configs, f'{run_config["estimator_name"].lower()}.yml')
    with open(path_sweep, 'r') as file:
        sweep = yaml.safe_load(file)
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep=sweep,
        entity=cst.entity,
        project=cst.project,
    )
    
    # Launch sweep
    if run_config['dry']:
        wandb.agent(
            sweep_id=sweep_id,
            function=train,
            entity=cst.entity,
            project=cst.project,
            count=5,
        ) 
    else:
        launch_sweep(nb_agents=run_config['nb_agents'], sweep_id=sweep_id)


def launch_sweep(nb_agents: int, sweep_id: str):
    list_agent = []
    for _ in range(nb_agents):
        agent = Process(
            target=wandb.agent,
            kwargs={
                'sweep_id': sweep_id,
                'function': train,
                'entity': cst.entity,
                'project': cst.project,
            }
        )
        list_agent.append(agent)
        agent.start()

    # complete the processes
    for agent in list_agent:
        agent.join()
    

def train():
    # Init wandb run
    run = wandb.init()
    # Get run config as dict
    run_config = run.config.as_dict()
    
    # Init pre-processor
    preprocessor = init_preprocessor(run_config)
    
    # Init target transformer
    transformer = init_transformer(run_config)
    
    # Init estimator
    estimator = init_estimator(run_config)
    
    # Pre-process data
    df_train = pd.read_csv(cst.file_data_train, index_col='ID')
    X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
    y_train = transformer.fit_transform(X_train, y_train)
    X_train = preprocessor.fit_transform(X_train)
    
    # Cross-validate estimator
    y_pred = cross_val_predict(
        estimator=estimator,
        X=X_train.to_numpy(),
        y=y_train.to_numpy(),
        cv=run_config['cv'],
        n_jobs=-1,
    )
    
    # Compute RMSE
    y_pred = transformer.inverse_transform(y_pred)
    rmse = mean_squared_error(y_pred=y_pred, y_true=y_train, squared=False)
    
    # Log results
    run.log({'rmse': rmse})
    
    # Finish run
    run.finish()


def parse_args() -> dict:
    # Define the parameters
    parser = argparse.ArgumentParser(description=f'Train {cst.project} model')
    parser.add_argument('--dry', action='store_true', default=False, help='Enable or disable dry mode pipeline')
    parser.add_argument('--estimator_name', type=str, default='XGBoost', choices=['XGBoost'], help='Estimator to use.')
    parser.add_argument('--nb_agents', type=int, default=1, help='Number of agents to run')
    return parser.parse_args().__dict__


if __name__ == '__main__':
    main()
