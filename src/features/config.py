from typing import Any

class CYEConfigPreProcessor:
    def __init__(
        self,
        fillna=True,
        missing_thr=50,
        fill_mode='median',
        target_name='Yield',
        *args: Any,
        **kwargs: Any
    ) -> None:
        
        self.fillna=fillna
        self.missing_thr=missing_thr
        self.fill_mode=fill_mode
        self.target_name=target_name