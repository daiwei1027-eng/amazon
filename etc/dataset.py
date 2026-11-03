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
        for i, series in enumerate(self.series):
            if series.height < self.lookback + self.horizon:
                continue
            for m in range(self.lookback, series.height - self.horizon + 1):
                self.sample_index.append((i,m))

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
        past_target = torch.tensor(
            history["units_sold"].to_numpy(),
            dtype=torch.float32,
        )

        target = torch.tensor(
            future["units_sold"].to_numpy(),
            dtype=torch.float32,
        )
        return past_target, target