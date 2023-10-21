import os
import GPUtil

from src.config import XGBConfig, LGBMConfig, LCEConfig
# from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from lce import LCERegressor


class CYEConstants:
    """
    Class to define constants and configurations for the FLAIR-2 project.
    """

    def __init__(self) -> None:
        # Paths to data directories
        self.path_data = os.path.join('data', 'raw')
        self.file_data_train = os.path.join(self.path_data, 'Train.csv')
        self.file_data_test = os.path.join(self.path_data, 'Test.csv')

        # Paths for models and submissions
        self.path_models = 'models'
        self.path_submissions = 'submissions'
        self.path_configs = 'configs'

        # Initialize the device for computation
        self.device = self.init_device()

        # WandB constants
        self.project = 'crop-yield-estimate'
        self.entity = 'association-rosia'

        self.target_column = 'Yield'

        self.estimators = {
            'XGBoost': {
                'config': XGBConfig,
                # 'estimator': XGBRegressor
            },
            'LightGBM': {
                'config': LGBMConfig,
                'estimator': LGBMRegressor
            },
            'LCE': {
                'config': LCEConfig,
                # 'estimator': LCERegressor
            }
        }

    @staticmethod
    def init_device():
        """
        Initialize device to work with.
        
        Returns:
            device (str): Device to work with (cpu, gpu).
        """
        device = 'cpu'
        if GPUtil.getAvailable():
            device = 'gpu'
        # * MPS is not implemented
        # elif mps.is_available():
        #     device = 'mps'

        return device


def get_constants() -> CYEConstants:
    """
    Get an instance of CYEConstants class with predefined constants and configurations.

    Returns:
        CYEConstants: Instance of CYEConstants class.
    """
    return CYEConstants()
