import argparse
import os
import sys

sys.path.append(os.curdir)

import pandas as pd
from pandas import Series

import wandb
from wandb.apis.public import Run

from src.models.utils import init_estimator, init_preprocessor, init_transformer

from src.constants import get_constants

cst = get_constants()


def main():
    # Get run id
    list_run_id = parse_args()

    list_predict = []
    for run_id in list_run_id:
        list_predict.append(predict(run_id))
    df_preds = pd.concat(list_predict, axis='columns', join='inner')

    # Create submisiion file to be uploaded to Zindi for scoring
    submission = df_preds.mean(axis='columns')
    submission.name = 'Yield'
    file_submission = os.path.join(cst.path_submissions, f'{"-".join(list_run_id)}.csv')
    submission.to_csv(file_submission, index=True)


def predict(run_id) -> Series:
    # Get run config
    run_config = get_run_config(run_id)

    # Init pre-processor
    preprocessor = init_preprocessor(run_config)

    # Init target transformer
    transformer = init_transformer(run_config)

    # Init estimator
    estimator = init_estimator(run_config)

    # Pre-process Train data
    df_train = pd.read_csv(cst.file_data_train, index_col='ID')
    X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
    y_train = transformer.fit_transform(X_train, y_train)
    X_train = preprocessor.fit_transform(X_train)

    # Train model
    estimator.fit(X=X_train.to_numpy(), y=y_train.to_numpy())

    # Pre-process Test data
    X_test = pd.read_csv(cst.file_data_test, index_col='ID')
    transformer.fit(X_test)
    X_test = preprocessor.transform(X_test)

    # Predict target value
    y_pred = estimator.predict(X=X_test.to_numpy())
    y_pred = Series(y_pred, index=X_test.index)
    y_pred = transformer.inverse_transform(y_pred)

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

    return parser.parse_args().run_id


if __name__ == '__main__':
    main()
