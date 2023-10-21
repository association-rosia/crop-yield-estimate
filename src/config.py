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
                 n_estimators: int = None,
                 max_depth: int = None,
                 max_leaves: int = None,
                 max_bin: int = None,
                 grow_policy: str = None,
                 learning_rate: float = None,
                 booster: str = None,
                 tree_method: str = None,
                 gamma: float = None,
                 min_child_weight: float = None,
                 max_delta_step: float = None,
                 subsample: float = None,
                 colsample_bytree: float = None,
                 colsample_bylevel: float = None,
                 colsample_bynode: float = None,
                 reg_alpha: float = None,
                 reg_lambda: float = None,
                 scale_pos_weight: float = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.grow_policy = grow_policy
        self.learning_rate = learning_rate
        self.booster = booster
        self.tree_method = tree_method
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight


class LGBMConfig(BaseConfig):
    def __init__(self,
                 boosting_type: str = None,
                 num_leaves: int = None,
                 max_depth: int = None,
                 learning_rate: float = None,
                 n_estimators: int = None,
                 subsample_for_bin: int = None,
                 min_split_gain: int = None,
                 min_child_weight: int = None,
                 min_child_samples: int = None,
                 subsample: float = None,
                 subsample_freq: int = None,
                 colsample_bytree: float = None,
                 reg_alpha: float = None,
                 reg_lambda: float = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda


class LCEConfig(BaseConfig):
    def __init__(self,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()

