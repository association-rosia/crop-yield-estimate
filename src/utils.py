from pandas import Series


def create_labels(y: Series, acre: Series, limit_h: int, limit_l: int) -> Series:
    # 0: Low, 1: Middle, 2: High
    target_by_acre = y / acre
    y_h = target_by_acre > limit_h
    y_h_m = target_by_acre > limit_l

    return y_h.astype(int) + y_h_m.astype(int)
