from sklearn.base import BaseEstimator
from src.features.config import CYEConfigPreProcessor, CYEConfigTransformer
from src.features.preprocessing import CYEDataPreProcessor, CYETargetTransformer

from src.constants import get_constants
cst = get_constants()


def init_preprocessor(run_config: dict) -> CYEDataPreProcessor:
    config_preprocessor = CYEConfigPreProcessor(**run_config)
    preprocessor = CYEDataPreProcessor(config_preprocessor)

    return preprocessor


def init_transformer(run_config: dict) -> CYETargetTransformer:
    config_transformer = CYEConfigTransformer(**run_config)

    return CYETargetTransformer(config_transformer)


def init_estimator(run_config: dict) -> BaseEstimator:
    estimator_name = run_config['estimator_name']
    estimator_config = cst.estimators[estimator_name]['config'](**run_config).get_params()
    estimator = cst.estimators[estimator_name]['estimator'](**estimator_config)

    return estimator
