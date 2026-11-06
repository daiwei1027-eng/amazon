from torch.utils.data import Dataset
import polars as pl
import torch
import numpy as np
class ForecastDataset(Dataset):
    def __init__(
        self,
        sales: pl.DataFrame,
        lookback: int,
        horizon: int,
            *,feature_columns: list[str] | None = None,
warmup: int = 0,
    ):
        self.lookback = lookback
        self.horizon = horizon

        self.series = (
            sales
            .sort(["store_id", "item_id", "date"])
            .partition_by(
                ["store_id", "item_id"],
                maintain_order=True,
            )
        )

        self.sample_index = []
        self.feature_columns = feature_columns
        first_target = self.lookback + warmup
        for i, series in enumerate(self.series):
            if series.height < first_target + self.horizon:
                continue
            for m in range(first_target, series.height - self.horizon + 1):
                self.sample_index.append((i, m))

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, index):
        if not 0 <= index < len(self.sample_index):
            raise IndexError(index)

        series_index, target_start = self.sample_index[index]
        series = self.series[series_index]


        history = series[
                  target_start - self.lookback: target_start
                  ]
        future = series[
                 target_start: target_start + self.horizon
                 ]
        if self.feature_columns is None:
            values = history["units_sold"].to_numpy()
        else:
            if "sell_price" in self.feature_columns:
                history = history.with_columns(
                    pl.col("sell_price").fill_null(0.0)
                )
            values = history.select(self.feature_columns).to_numpy()

        past_target = torch.tensor(values, dtype=torch.float32)

        target = torch.tensor(
            future["units_sold"].to_numpy(),
            dtype=torch.float32,
        )
        return past_target, target