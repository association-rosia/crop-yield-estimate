from sklearn.base import RegressorMixin, ClassifierMixin
from src.features.config import CYEConfigPreProcessor, CYEConfigTransformer
from src.features.preprocessing import CYEDataPreProcessor, CYETargetTransformer
from src.models.custom_model import CustomEstimator

from wandb.plot import confusion_matrix

from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


from src.constants import get_constants
cst = get_constants()


def init_preprocessor(run_config: dict) -> CYEDataPreProcessor:
    config_preprocessor = CYEConfigPreProcessor(**run_config)
    preprocessor = CYEDataPreProcessor(config_preprocessor)

    return preprocessor


def init_transformer(run_config: dict) -> CYETargetTransformer:
    config_transformer = CYEConfigTransformer(**run_config)

    return CYETargetTransformer(config_transformer)


def init_estimator(run_config: dict) -> RegressorMixin | ClassifierMixin:
    estimator_name = run_config['estimator_name']

    if run_config['task'] == 'regression':
        estimators = cst.reg_estimators
    elif run_config['task'] == 'classification':
        estimators = cst.cls_estimators

    estimator_config = estimators[estimator_name]['config'](**run_config).get_params()
    estimator = estimators[estimator_name]['estimator'](**estimator_config)

    return estimator


def regression_metrics(y_pred, y_true) -> dict:
    metrics = {
        'reg/rmse': mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)
    }
    
    return metrics


def classication_metrics(y_pred, y_true) -> dict:
    metrics = {
        'cls/accuracy': accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True),
        'cls/f1': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
        'cls/recall': recall_score(y_true=y_true, y_pred=y_pred, average='macro'),
        'cls/precision': precision_score(y_true=y_true, y_pred=y_pred, average='macro'),
        'cls/confusion_matrix': confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=['Low', 'Medium', 'High'],
            title='Confusion matrix',
            )
    }
    
    return metrics

def init_evaluation_metrics(run_config: dict):
    if run_config['task'] == 'regression':
        evaluation_metrics = regression_metrics
    elif run_config['task'] == 'classification':
        evaluation_metrics = classication_metrics
    
    return evaluation_metrics