import os
import sys
from typing import Any

sys.path.append(os.curdir)

from src.config import BaseConfig


class CYEConfigPreProcessor(BaseConfig):
    def __init__(
            self,
            delna_thr=50,
            fillna=False,
            normalisation=False,
            scale='none',
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__()

        self.delna_thr = delna_thr
        self.fillna = fillna
        self.normalisation = normalisation
        self.scale = scale


class CYEConfigTransformer(BaseConfig):
    def __init__(
            self,
            scale='none',
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__()

        self.scale = scale
