import os

import GPUtil
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from src.config import XGBConfig, LGBMConfig


class CYEConstants:
    """
    Class to define constants and configurations for the FLAIR-2 project.
    """

    def __init__(self) -> None:
        # Paths to data directories
        self.path_interim_data = os.path.join('data', 'interim') if os.path.exists(os.path.join('data', 'interim')) else os.path.join('..', 'data', 'interim')
        self.path_generated_data = os.path.join('data', 'generated') if os.path.exists(os.path.join('data', 'generated')) else os.path.join('..', 'data', 'generated')
        self.path_raw_data = os.path.join('data', 'raw') if os.path.exists(os.path.join('data', 'raw')) else os.path.join('..', 'data', 'raw')

        self.file_data_train = os.path.join(self.path_raw_data, 'Train.csv')
        self.file_data_test = os.path.join(self.path_raw_data, 'Test.csv')

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

        self.processor = self.init_processor_constants()
        self.outliers_thr = self.init_outliers_thr()
        self.reg_estimators = self.init_reg_estimators()
        self.cls_estimators = self.init_cls_estimators()

    @staticmethod
    def init_processor_constants():
        processor = {
            'list_cols': ['LandPreparationMethod', 'NursDetFactor', 'TransDetFactor', 'OrgFertilizers',
                          'CropbasalFerts', 'FirstTopDressFert'],
            'date_cols': ['CropTillageDate', 'RcNursEstDate', 'SeedingSowingTransplanting', 'Harv_date',
                          'Threshing_date'],
            'cat_cols': ['District', 'Block', 'CropEstMethod', 'TransplantingIrrigationSource',
                         'TransplantingIrrigationPowerSource', 'PCropSolidOrgFertAppMethod', 'MineralFertAppMethod',
                         'MineralFertAppMethod.1', 'Harv_method', 'Threshing_method', 'Stubble_use'],
            'corr_list_cols': [('Ganaura', 'OrgFertilizersGanaura'), ('BasalUrea', 'CropbasalFertsUrea'),
                               ('1tdUrea', 'FirstTopDressFertUrea')],
            'to_del_cols': ['Harv_methodmachine', 'OrgFertilizersJeevamrit', 'CropbasalFertsMoP',
                            'FirstTopDressFertNPK', 'TransplantingIrrigationSourceWell',
                            'TransplantingIrrigationPowerSourceSolar', 'Harv_dateYear2022', 'Harv_dateYear2023'],
            'corr_area_cols': ['CultLand', 'CropCultLand', 'TransIrriCost', 'Ganaura', 'CropOrgFYM', 'Harv_hand_rent',
                               'BasalUrea', '1tdUrea', '2tdUrea']
        }

        processor['cat_cols'] += [f'{col}Year' for col in processor['date_cols']]

        return processor

    @staticmethod
    def init_outliers_thr():
        outliers_thr = {
            'CultLand': 200,
            'CropCultLand': 200,
            'SeedlingsPerPit': 20,
            'TransplantingIrrigationHours': 1000,
            'TransIrriCost': 3000,
            'Ganaura': 900,
            '1appDaysUrea': 65,
            'Harv_dateDayOfYearSin': 0,
        }

        return outliers_thr

    @staticmethod
    def init_reg_estimators():
        reg_estimators = {
            'XGBoost': {
                'config': XGBConfig,
                'estimator': XGBRegressor
            },
            'LightGBM': {
                'config': LGBMConfig,
                'estimator': LGBMRegressor
            },
            # 'LCE': {
            #     'config': LCEConfig,
            #     'estimator': LCERegressor
            # }
        }

        return reg_estimators

    @staticmethod
    def init_cls_estimators():
        cls_estimators = {
            'XGBoost': {
                'config': XGBConfig,
                'estimator': XGBClassifier
            },
            'LightGBM': {
                'config': LGBMConfig,
                'estimator': LGBMClassifier
            },
        }

        return cls_estimators

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
