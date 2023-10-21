from sklearn.base import BaseEstimator
from xgboost.sklearn import XGBRegressor

from src.config import XGBConfig
from src.features.config import CYEConfigPreProcessor
from src.features.preprocessing import CYEDataPreProcessor


def init_preprocessor(run_config: dict) -> CYEDataPreProcessor:
    config_preprocessor = CYEConfigPreProcessor(**run_config)

    return CYEDataPreProcessor(config_preprocessor)


def init_estimator(run_config: dict) -> BaseEstimator:
    if run_config['estimator_name'] == 'XGBoost':
        estimator_config = XGBConfig(**run_config)
        estimator = XGBRegressor(**estimator_config.get_params())
    else:
        raise NotImplementedError(f'Estimator {run_config["estimator_name"]} are not implemented.')

    return estimator
