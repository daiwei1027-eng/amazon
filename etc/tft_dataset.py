import polars as pl
import torch
from etc.dataset import ForecastDataset


class TFTDataset(ForecastDataset):
    def __init__(
        self,
        sales: pl.DataFrame,
        lookback: int,
        horizon: int,
        *,
        history_observed_features: list[str],
        known_future_features: list[str],
        static_mappings: dict[str, dict[str, int]],
        warmup: int = 28,
    ):
        super().__init__(
            sales,
            lookback,
            horizon,
            warmup=warmup,
        )
        self.history_observed_features = list(history_observed_features)
        self.known_future_features = list(known_future_features)
        self.static_mappings = static_mappings
        self.static_features = list(static_mappings)
    def __getitem__(self, index):
        if not 0 <= index < len(self.sample_index):
            raise IndexError(index)

        series_index, target_start = self.sample_index[index]
        series = self.series[series_index]

        history = series[
                  target_start - self.lookback:target_start
                  ]
        future = series[
                 target_start:target_start + self.horizon
                 ]

        if "sell_price" in self.history_observed_features:
            history = history.with_columns(
                pl.col("sell_price").fill_null(0.0)
            )

        inputs = {
            "past_observed": torch.tensor(
                history.select(self.history_observed_features).to_numpy(),
                dtype=torch.float32,
            ),
            "past_known": torch.tensor(
                history.select(self.known_future_features).to_numpy(),
                dtype=torch.float32,
            ),
            "future_known": torch.tensor(
                future.select(self.known_future_features).to_numpy(),
                dtype=torch.float32,
            ),
            "static": torch.tensor(
    [
        self.static_mappings[name][series[name][0]]
        for name in self.static_features
    ],
    dtype=torch.long,
            ),
        }

        target = torch.tensor(
            future["units_sold"].to_numpy(),
            dtype=torch.float32,
        )

        return inputs, target