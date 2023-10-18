import argparse
import os
import sys
from argparse import Namespace

sys.path.append(os.curdir)

import pandas as pd
import wandb

from wandb.sdk import wandb_config, wandb_run

from src.features.preprocessing import CYEDataPreProcessor
from src.features.config import CYEConfigPreProcessor

from xgboost.sklearn import XGBRegressor

from src.constants import get_constants

cst = get_constants()


def main():
    # Configure & Start run
    args_config = parse_args()
    run = init_wandb(args_config)

    # Prepare model directory
    name_run = f'{run.name}-{run.id}'
    path_model = os.path.join(cst.path_models, name_run)
    os.makedirs(path_model)

    # Init pre-processing pipeline 
    preprocessor = init_preprocessor(run.config)

    # Pre-process data
    raw_data = pd.read_csv(cst.file_data_train)
    input_data = preprocessor.preprocess(raw_data)
    input_data = preprocessor.fit_transform(input_data)

    # Save pre-processing pipeline
    file_preprocessor = os.path.join(path_model, 'preprocessor.json')
    preprocessor.save_dict(file_preprocessor)

    # Init model
    xgbr = XGBRegressor()

    # Train model
    xgbr.fit(input_data.drop(columns='Yield'), input_data['Yield'])

    # Save model
    file_model = os.path.join(path_model, 'model.json')
    xgbr.save_model(file_model)

    # Finish run
    run.finish()


def init_preprocessor(run_config: wandb_config.Config | dict) -> CYEDataPreProcessor:
    config_preprocessor = CYEConfigPreProcessor(**run_config)

    return CYEDataPreProcessor(config_preprocessor)


def init_wandb(args_config: Namespace | dict) -> wandb_run.Run:
    run = wandb.init(
        entity=cst.entity,
        project=cst.project_name,
        config=args_config
    )

    return run


def parse_args() -> Namespace:
    # Define the parameters
    parser = argparse.ArgumentParser(description=f'Train {cst.project_name} model')
    parser.add_argument('--dry', action='store_true', default=False, help='Enable or disable dry mode pipeline')

    # Pre Processing 
    parser.add_argument('--fillna', action='store_true', default=True,
                        help='Fill or not missing values during preprocessing')
    parser.add_argument('--missing_thr', type=int, default=50,
                        help='Threshold in percentage (0 - 100). Delete columns that have more missing values than it')
    parser.add_argument('--fill_mode', type=str, default='median', choices=['median', 'mean'],
                        help='Methode used to fill missing values')

    return parser.parse_args()


if __name__ == '__main__':
    main()
