import os
import sys
from typing import Any

sys.path.append(os.curdir)

from src.config import BaseConfig


class CYEConfigPreProcessor(BaseConfig):
    def __init__(self,
                 delna_thr=50,
                 fill_mode='none',
                 normalisation=True,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.delna_thr = delna_thr
        self.fill_mode = fill_mode
        self.normalisation = normalisation
