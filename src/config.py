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

    def get_params(self, deep=True):
        """
        Get parameters for this configuration.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out


class XGBConfig(BaseConfig):
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


class LGBMConfig(BaseConfig):
    def __init__(self,
                 metric: str = None,
                 device: str = None,
                 n_estimators: int = None,
                 learning_rate: float = None,
                 max_depth: int = None,
                 min_data_in_leaf: int = None,
                 num_leaves: int = None,
                 subsample: float = None,
                 colsample_bytree: float = None,
                 reg_lambda: float = None,
                 reg_alpha: float = None,
                 gamma: float = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.metric = metric
        self.device = device
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma


class LCEConfig(BaseConfig):
    def __init__(self,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
