from typing import Any
from src.config import BaseConfig


def check_scale(value: str) -> str:
    value_option = ['none', 'Acre', 'CultLand', 'CropCultLand']
    if value not in value_option:
        raise ValueError(f'scale can be {", ".join(value_option)}, but found {value}')

    return value


def check_task(value: str) -> str:
    value_option = ['regression', 'classification', 'reg_l', 'reg_m', 'reg_h']
    if value not in value_option:
        raise ValueError(f'task can be {", ".join(value_option)}, but found {value}')

    return value


def check_fillna(value: str | bool) -> str:
    value_option = ['none', 'KNNImputer', 'IterativeImputer']
    
    # Backward compatibility
    if isinstance(value, bool) and value:
        value = 'KNNImputer'
    # Backward compatibility
    elif isinstance(value, bool) and not value:
        value = 'none'
    elif value not in value_option:
        raise ValueError(f'fillna can be {", ".join(value_option)}, but found {value}')

    return value


def check_deloutliers(value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'deloutliers must be a boolean, but found {type(value)}')

    return value


def check_delna_thr(value: float) -> float:
    if value > 1 or value < 0:
        raise ValueError(f'delna_thr must be between 0 and 1, but found {value}')

    return value


def check_yieldoutliers_thr(value: float | None) -> str:
    if value is not None and value <= 0:
        raise ValueError(f'yieldoutliers_thr must be positive, but found {value}')

    return value


def check_limit_h(value):
    return value


def check_limit_l(value):
    return value


class CYEConfigPreProcessor(BaseConfig):
    def __init__(
            self,
            delna_thr=1,
            fillna='none',
            deloutliers=False,
            scale='none',
            yieldoutliers_thr=None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__()

        self.delna_thr = check_delna_thr(delna_thr)
        self.fillna = check_fillna(fillna)
        self.deloutliers = check_deloutliers(deloutliers)
        self.scale = check_scale(scale)
        self.yieldoutliers_thr = check_yieldoutliers_thr(yieldoutliers_thr)


class CYEConfigTransformer(BaseConfig):
    def __init__(
            self,
            scale='none',
            task='regression',
            limit_h=None,
            limit_l=None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__()

        self.scale = check_scale(scale)
        self.task = check_task(task)
        self.limit_h = check_limit_h(limit_h)
        self.limit_l = check_limit_l(limit_l)
        self.check_parameters()

    def check_parameters(self):
        if self.task in ['classification', 'reg_l', 'reg_m', 'reg_h']:
            if self.limit_h is None:
                raise ValueError(f'For {self.task} task, limit_h must be defined, found None.')
            if self.limit_l is None:
                raise ValueError(f'For {self.task} task, limit_l must be defined, found None.')

        if self.task == 'classififcation':
            if self.scale != 'none':
                raise ValueError(f'For {self.task} task, scale must be equal to \'none\', found {self.scale}.')
