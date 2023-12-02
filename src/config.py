import inspect
from typing import Any


class BaseConfig:
    def __init__(self) -> None:
        pass

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the configuration"""
        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(cls.__init__)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD and p.kind != p.VAR_POSITIONAL
        ]

        return sorted([p.name for p in parameters])

    def get_params(self):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if value is not None:
                out[key] = value
        return out


class XGBoostConfig(BaseConfig):
    def __init__(self,
                 eval_metric: str = None,
                 n_estimators: int = None,
                 learning_rate: float = None,
                 max_depth: int = None,
                 subsample: float = None,
                 colsample_bytree: float = None,
                 colsample_bylevel: float = None,
                 colsample_bynode: float = None,
                 min_child_weight: float = None,
                 reg_lambda: float = None,
                 reg_alpha: float = None,
                 gamma: float = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.eval_metric = eval_metric
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma


class LightGBMConfig(BaseConfig):
    def __init__(self,
                 metric: str = None,
                 n_estimators: int = None,
                 learning_rate: float = None,
                 max_depth: int = None,
                 min_child_samples: int = None,
                 num_leaves: int = None,
                 subsample: float = None,
                 colsample_bytree: float = None,
                 reg_lambda: float = None,
                 reg_alpha: float = None,
                 verbose: int = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.metric = metric
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.verbose = verbose


class CatBoostConfig(BaseConfig):
    def __init__(self,
                 loss_function: str = None,
                 n_estimators: int = None,
                 learning_rate: float = None,
                 reg_lambda: float = None,
                 bootstrap_type: str = None,
                 subsample: float = None,
                 max_depth: int = None,
                 grow_policy: str = None,
                 leaf_estimation_iterations: int = None,
                 leaf_estimation_backtracking: str = None,
                 auto_class_weights: str = None,
                 colsample_bylevel: float = None,
                 nan_mode: str = None,
                 langevin: bool = None,
                 diffusion_temperature: int = None,
                 score_function: str = None,
                 penalties_coefficient: float = None,
                 model_shrink_rate: float = None,
                 model_shrink_mode: str = None,
                 verbose: int = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.loss_function = loss_function
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.max_depth = max_depth
        self.grow_policy = grow_policy
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.leaf_estimation_backtracking = leaf_estimation_backtracking
        if loss_function != 'RMSE':
            self.auto_class_weights = auto_class_weights
        self.colsample_bylevel = colsample_bylevel
        self.nan_mode = nan_mode
        self.langevin = langevin
        self.diffusion_temperature = diffusion_temperature
        self.score_function = score_function
        self.penalties_coefficient = penalties_coefficient
        self.model_shrink_rate = model_shrink_rate
        self.model_shrink_mode = model_shrink_mode
        self.verbose = verbose
