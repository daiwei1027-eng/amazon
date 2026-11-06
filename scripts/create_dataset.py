import polars as pl
from torch.utils.data import Dataset

from etc.dataset import ForecastDataset


def create_dataset(
    sales: pl.DataFrame,lookback: int,horizon: int,*,feature_columns: list[str] | None = None,warm_up: int = 0,
)->Dataset:
    for name, value in [
        ("lookback", lookback),
        ("horizon", horizon),
    ]:
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    return ForecastDataset(sales,lookback,horizon,feature_columns=feature_columns,warmup=warm_up)
