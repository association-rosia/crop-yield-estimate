import argparse
import os
import sys

sys.path.append(os.curdir)

import pandas as pd
from pandas import DataFrame, Series

import wandb
from wandb.apis.public import Run

from src.models.utils import (
    init_estimator,
    init_preprocessor,
    init_target_transformer,
    get_train_data,
    get_test_data,
    apply_smote,
    apply_great
)

from src.constants import get_constants

cst = get_constants()


def main():
    # Get run id
    predict_config = parse_args()
    
    # Init ensemble strategy
    make_ensemble = init_ensemble_strategy(predict_config)

    list_predict = []
    for run_id in predict_config['run_id']:
        list_predict.append(predict(run_id))
    df = pd.concat(list_predict, axis='columns', join='inner')
    
    # Apply ensemble strategy
    submission = make_ensemble(df)
    
    # Create submisiion file to be uploaded to Zindi for scoring
    submission.name = 'Yield'
    file_submission = os.path.join(cst.path_submissions, f'{"-".join(predict_config["run_id"])}.csv')
    submission.to_csv(file_submission, index=True)
    
    return True


def classification_strategy(df: DataFrame) -> Series:
    # Classification ensemble strategy
    def get_value(row):
        if row[0] == 0:
            return row[1]
        elif row[0] == 1:
            return row[2]
        elif row[0] == 2:
            return row[3]
        
    return df.apply(get_value, axis=1)


def mean_strategy(df: DataFrame) -> Series:
    return df.mean(axis='columns')


def init_ensemble_strategy(predict_config: dict):
    if predict_config['ensemble_strategy'] == 'mean':
        ensemble_strategy = mean_strategy
    if predict_config['ensemble_strategy'] == 'classification':
        ensemble_strategy = classification_strategy
        
    return ensemble_strategy


def predict(run_id) -> Series:
    # Get run config
    run_config = get_run_config(run_id)

    # Init pre-processor
    preprocessor = init_preprocessor(run_config)

    # Init target transformer
    target_transformer = init_target_transformer(run_config)

    # Init estimator
    estimator = init_estimator(run_config)

    # Load train data
    df_train = get_train_data(run_config)
    
    # Pre-process Train data
    X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
    y_train = target_transformer.fit_transform(X_train, y_train)
    X_train = preprocessor.fit_transform(X_train)

    # Train model
    if run_config['great_augmentation']:
        X_train, y_train = apply_great(run_config, X_train, y_train, target_transformer, preprocessor)

    if run_config['smote_augmentation']:
        X_train, y_train = apply_smote(X_train, y_train)

    estimator.fit(X=X_train.to_numpy(), y=y_train.to_numpy())

    # Load test data
    X_test = get_test_data()
    
    # Pre-process Test data
    target_transformer.fit(X_test)
    X_test = preprocessor.transform(X_test)

    # Predict target value
    y_pred = estimator.predict(X=X_test.to_numpy())
    y_pred = Series(y_pred, index=X_test.index)
    y_pred = target_transformer.inverse_transform(y_pred)

    return y_pred


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
    parser.add_argument('--run_id', nargs='+', type=str,
                        help='ID of wandb run to use for submission. Give multiple IDs for ensemble submission.')
    
    parser.add_argument('--ensemble_strategy', type=str, default='mean', choices=['mean', 'classification'], 
                        help='Ensemble strategy to use. If classification is choised, the task runs must be '
                             'classification, reg_low, reg_medium, reg_high')

    return parser.parse_args().__dict__


if __name__ == '__main__':
    main()
