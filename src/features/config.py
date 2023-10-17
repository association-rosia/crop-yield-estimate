from typing import Any


class CYEConfigPreProcessor:
    def __init__(
            self,
            fillna=True,
            delna_thr=50,
            fill_mode='median',
            *args: Any,
            **kwargs: Any
    ) -> None:
        self.fillna = fillna
        self.delna_thr = delna_thr
        self.fill_mode = fill_mode
