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
    init_transformer,
    get_train_data,
    get_test_data,
    apply_smote
)

from src.constants import get_constants

cst = get_constants()


def main():
    # Get run id
    predict_config = parse_args()
    
    # Init ensemble strategy
    make_ensemble = init_ensemble_strategy(predict_config)

    # Predict 
    class_pred = get_class_prediction(predict_config)

    list_predict = get_prediction(predict_config['low_id'], 0, class_pred)
    list_predict += get_prediction(predict_config['medium_id'], 1, class_pred)
    list_predict += get_prediction(predict_config['high_id'], 2, class_pred)
    list_predict += get_prediction(predict_config['run_id'])
    
    df = pd.concat(list_predict, axis='columns', join='outer')
    
    # Apply ensemble strategy
    submission = make_ensemble(df)
    
    # Create submisiion file to be uploaded to Zindi for scoring
    submission.name = 'Yield'
    list_id = predict_config['run_id'] + predict_config['low_id'] +  predict_config['medium_id'] + predict_config['high_id']
    file_name =  f'{"-".join(list_id)}-{predict_config["ensemble_strategy"]}.csv'
    file_submission = os.path.join(cst.path_submissions, file_name)
    submission.to_csv(file_submission, index=True)
    
    return True


def get_class_prediction(predict_config: dict) -> Series:
    list_predict_class = []
    for class_id in predict_config['class_id']:
        list_predict_class.append(predict(class_id))
    df = pd.concat(list_predict_class, axis='index')
    class_prediction = df.groupby('ID').median()
    class_prediction = class_prediction.idxmax(axis='columns')
    
    return class_prediction


def get_prediction(run_ids: list, class_idx: int = None, class_pred: Series = None) -> list:
    list_predict = []
    for run_id in run_ids:
        y_pred = predict(run_id)
        if class_idx is not None and class_pred is not None:
            y_pred = y_pred.loc[class_pred[class_pred == class_idx].index]
        list_predict.append(y_pred)

    return list_predict


def mean_strategy(df: DataFrame) -> Series:
    return df.mean(axis='columns')


def median_strategy(df: DataFrame) -> Series:
    return df.median(axis='columns')


def init_ensemble_strategy(predict_config: dict):
    if predict_config['ensemble_strategy'] == 'mean':
        ensemble_strategy = mean_strategy
    if predict_config['ensemble_strategy'] == 'median':
        ensemble_strategy = median_strategy
        
    return ensemble_strategy


def predict(run_id) -> Series | DataFrame:
    # Get run config
    run_config = get_run_config(run_id)

    # Init pre-processor
    preprocessor = init_preprocessor(run_config)

    # Init target transformer
    transformer = init_transformer(run_config)

    # Init estimator
    estimator = init_estimator(run_config)

    # Load train data
    df_train = get_train_data(run_config)
    
    # Pre-process Train data
    X_train, y_train = df_train.drop(columns=cst.target_column), df_train[cst.target_column]
    y_train = transformer.fit_transform(X_train, y_train)
    X_train = preprocessor.fit_transform(X_train)

    # Train model
    if 'data_aug' in run_config and run_config['data_aug'] == 'smote':
        X_train, y_train = apply_smote(X_train, y_train)

    estimator.fit(X=X_train.to_numpy(), y=y_train.to_numpy())

    # Load test data
    X_test = get_test_data()
    
    # Pre-process Test data
    transformer.fit(X_test)
    X_test = preprocessor.transform(X_test)

    # Predict target value
    if run_config['task'] == 'classification':
        y_pred = estimator.predict_proba(X=X_test.to_numpy())
        y_pred = DataFrame(y_pred, index=X_test.index)
    else:
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
    parser.add_argument('--run_id', nargs='+', type=str, default=[],
                        help='ID of wandb run to use for submission. Give multiple IDs for ensemble submission.')
    
    parser.add_argument('--class_id', nargs='+', type=str, default=[],
                        help='ID of wandb run trained to classify data to use for submission. Give multiple IDs for ensemble submission.')
    
    parser.add_argument('--low_id', nargs='+', type=str, default=[],
                        help='ID of wandb run trained on low data to use for submission. Give multiple IDs for ensemble submission.')
    
    parser.add_argument('--medium_id', nargs='+', type=str, default=[],
                        help='ID of wandb run trained on medium data to use for submission. Give multiple IDs for ensemble submission.')
    
    parser.add_argument('--high_id', nargs='+', type=str, default=[],
                        help='ID of wandb run trained on high data to use for submission. Give multiple IDs for ensemble submission.')
    
    parser.add_argument('--ensemble_strategy', type=str, default='mean', choices=['mean', 'median'], 
                        help='Ensemble strategy to use. If classification is choised, the task runs must be '
                             'classification, reg_low, reg_medium, reg_high')

    return parser.parse_args().__dict__


if __name__ == '__main__':
    main()
