from typing import Any


class CYEConfigPreProcessor:
    def __init__(
            self,
            fillna=True,
            delna_thr=50,
            fill_mode='median',
            normalisation=True,
            target_name='Yield',
        *args: Any,
            **kwargs: Any
    ) -> None:
        self.fillna = fillna
        self.delna_thr = delna_thr
        self.fill_mode = fill_mode
        self.normalisation = normalisation

        self.target_name=target_name