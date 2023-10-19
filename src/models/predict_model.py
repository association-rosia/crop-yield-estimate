import argparse
import os
import sys

sys.path.append(os.curdir)

import pandas as pd

import wandb
from wandb.apis.public import Run

from src.models.utils import init_estimator, init_preprocessor

from src.constants import get_constants

cst = get_constants()


def main():
    # Get run id
    run_id = parse_args()
    
    # Get run config
    run_config = get_run_config(run_id)
    
    # Init pre-processor
    preprocessor = init_preprocessor(run_config)
    
    # Init estimator
    estimator = init_estimator(run_config)
    
    # Pre-process Train data
    df_train = pd.read_csv(cst.file_data_train)
    X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
    X_train = preprocessor.fit_transform(X_train)
    
    # Train model
    estimator.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
    
    # Pre-process Test data
    X_test = pd.read_csv(cst.file_data_test)
    X_test = preprocessor.transform(X_test)
    
    estimator.predict(X_test)

    # Predict target value
    preds = estimator.predict(X=X_test.to_numpy())

    # Create submisiion file to be uploaded to Zindi for scoring
    submission = pd.DataFrame({'ID': X_test.index, 'Yield': preds})
    file_submission = os.path.join(cst.path_submissions, f'{run_id}.csv')
    submission.to_csv(file_submission, index=False)
    

def get_run_config(run_id: str) -> dict:
    api = wandb.Api()
    run = Run(
        client=api.client,
        entity=cst.entity,
        project=cst.project,
        run_id=run_id,
    )
    
    return run.config


def parse_args() -> dict:
    # Define the parameters
    parser = argparse.ArgumentParser(description=f'Make {cst.project} submission')

    # Run name
    parser.add_argument('--run_id', type=str, help='ID of wandb run to use for submission')

    return parser.parse_args().run_id

if __name__ == '__main__':
    main()
