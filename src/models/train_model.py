import argparse
import os
import sys
from multiprocessing import Process

import numpy as np
import yaml

sys.path.append(os.curdir)

import wandb

from src.models.utils import (
    init_preprocessor,
    init_estimator,
    init_transformer,
    init_cross_validator,
    init_evaluation_metrics,
    get_train_data,
    init_trainer
)

from src.constants import get_constants

cst = get_constants()


def main():
    # Configure & Start run
    script_config = parse_args()

    if script_config['debug']:
        wandb.init(
            config=os.path.join(cst.path_configs, 'debug-config.yml'),
            entity=cst.entity,
            project=cst.project,
        )
        train()
    else:
        sweep_id = create_sweep(script_config)
        # Launch sweep
        if script_config['dry']:
            wandb.agent(
                sweep_id=sweep_id,
                function=train,
                entity=cst.entity,
                project=cst.project,
                count=3
            )
        else:
            launch_sweep(nb_agents=script_config['nb_agents'], sweep_id=sweep_id)


def create_sweep(script_config: dict) -> str:
    # Load sweep config
    if script_config['task'] == 'regression':
        dir_config = 'regressors'
    elif script_config['task'] == 'classification':
        dir_config = 'classifiers'

    path_sweep = os.path.join(cst.path_configs, dir_config, f'{script_config["estimator_name"].lower()}.yml')
    with open(path_sweep, 'r') as file:
        sweep = yaml.safe_load(file)

    # Create sweep
    sweep_id = wandb.sweep(
        sweep=sweep,
        entity=cst.entity,
        project=cst.project,
    )

    return sweep_id


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

    # Init cross-validator generator
    cv = init_cross_validator(run_config)

    # Init evaluation metrics
    evaluation_metrics = init_evaluation_metrics(run_config)

    # Load train data
    df_train = get_train_data(run_config)  # for cls, the classes are calculate in transformer

    # Pre-process data
    X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
    y_train = transformer.fit_transform(X_train, y_train)
    X_train = preprocessor.fit_transform(X_train)

    # Cross-validate estimator
    trainer = init_trainer(run_config)
    y_pred = trainer(
        estimator=estimator,
        X=X_train.to_numpy(),
        y=y_train.to_numpy(),
        cv=cv,
        n_jobs=-1,
    )

    # Compute RMSE
    y_pred = transformer.inverse_transform(y_pred)
    y_train = transformer.inverse_transform(y_train)
    metrics = evaluation_metrics(y_pred=y_pred, y_true=y_train)

    # Log results
    run.log(metrics)

    # Finish run
    run.finish()

    return True


def parse_args() -> dict:
    # Define the parameters
    parser = argparse.ArgumentParser(description=f'Train {cst.project} model')
    parser.add_argument('--dry', action='store_true', default=False, help='Enable or disable dry mode pipeline')
    parser.add_argument('--debug', action='store_true', default=True,
                        help='Run a single training using debug-config.yml (for debug purpose)')
    parser.add_argument('--estimator_name', type=str, default='XGBoost', choices=['XGBoost', 'LightGBM'],
                        help='Estimator to use')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'],
                        help='Task to be performed')
    parser.add_argument('--nb_agents', type=int, default=1, help='Number of agents to run')

    return parser.parse_args().__dict__


if __name__ == '__main__':
    main()
