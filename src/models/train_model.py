import os
import sys
import argparse
import joblib

sys.path.append(os.curdir)

import pandas as pd

import wandb
from wandb.sdk import wandb_run
from wandb import Swe

from src.features.preprocessing import CYEDataPreProcessor
from src.features.config import CYEConfigPreProcessor

from src.models.model_config import get_gridsearch, get_parameters

from src.constants import get_constants

cst = get_constants()

def main():
    # Configure & Start run
    run_config = parse_args()
    if run_config['sweep']:
        sweep_id = wandb.sweep(
            sweep=f'{run_config["estimator_name"]}.yml',
            entity=cst.entity,
            project=cst.project_name,
        )
        
        wandb.agent(
            sweep_id=sweep_id,
            function=train,
            entity=cst.entity,
            project=cst.project_name,
        )
    else:
        train(run_config)

def train(run_config: dict | None):
    run = init_wandb(run_config)
    run_config = run.config.as_dict()
    
    # Prepare model directory
    name_run = f'{run.name}-{run.id}'
    path_model = os.path.join(cst.path_models, name_run)
    os.makedirs(path_model)
    
    # Init pre-processing pipeline
    preprocessor = init_preprocessor(run_config)
    raw_data = pd.read_csv(cst.file_data_train)
    # Pre-process data
    input_data = preprocessor.fit_transform(raw_data)
    # Save pre-processing pipeline
    file_preprocessor = os.path.join(path_model, 'preprocessor.json')
    preprocessor.save_dict(file_preprocessor)
    
    # Init GridSearch
    # gridsearch = get_gridsearch(run_config=run_config, target=input_data[run_config['target_name']])
    
    # Begin GridSearch & Save score
    gridsearch.fit(input_data.drop(columns='Yield').to_numpy(), input_data['Yield'].to_numpy())
    run.log({'loss': gridsearch.best_score_})
    
    # Save model
    file_model = os.path.join(path_model, 'model.save')
    joblib.dump(gridsearch.best_estimator_, filename=file_model)
    # Finish run
    run.finish()
    

def init_preprocessor(run_config: dict) -> CYEDataPreProcessor:
    config_preprocessor = CYEConfigPreProcessor(**run_config)
    
    return CYEDataPreProcessor(config_preprocessor)


def init_wandb(args_config: dict) -> wandb_run.Run:
    run = wandb.init(
        entity=cst.entity,
        project=cst.project_name,
        config=args_config
    )
    
    return run


def parse_args() -> dict:
    # Define the parameters
    parser = argparse.ArgumentParser(description=f'Train {cst.project_name} model')
    parser.add_argument('--dry', action='store_true', default=False, help='Enable or disable dry mode pipeline')
    parser.add_argument('--seed', type=int, default=42, help='Pass an int for reproducible output across multiple function calls.')
    
    # Pre Processing 
    parser.add_argument('--fillna', action='store_true', default=False, help='Fill or not missing values during preprocessing')
    parser.add_argument('--missing_thr', type=int, default=50, help='Threshold in percentage (0 - 100). Delete columns that have more missing values than it')
    parser.add_argument('--fill_mode', type=str, default='median', choices=['median', 'mean'], help='Methode used to fill missing values')
    parser.add_argument('--target_name', type=str, default='Yield', choices=['Yield'], help='Columns to use as target value')
    
    # Kfold
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds. Must be at least 2.')
    parser.add_argument('--estimator_name', type=str, default='XGBoost', choices=['XGBoost'], help='Estimator to use.')
    
    # Sweep
    parser.add_argument('--sweep', action='store_true', default=False, help="Use or not a sweep")
    
    return parser.parse_args().__dict__


if __name__ == '__main__':
    main()