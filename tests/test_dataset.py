from datetime import date, timedelta

import polars as pl
import pytest
import torch

from etc.dataset import ForecastDataset
from scripts.create_dataset import create_dataset


@pytest.fixture
def sales_frame():
    rows = []
    for item_id, base, price in [("A", 10, 2.5), ("B", 100, 5.0)]:
        for day in range(8):
            rows.append({
                "store_id": "STORE_1",
                "item_id": item_id,
                "date": date(2020, 1, 1) + timedelta(days=day),
                "units_sold": base + day,
                "sell_price": None if day < 3 else price,
                "is_weekend": day % 2 == 0,
                "lag_1": None if day == 0 else base + day - 1,
            })
    return pl.DataFrame(rows).reverse()


def test_default_history_and_sales_labels_are_sorted_per_series(sales_frame):
    dataset = ForecastDataset(sales_frame, lookback=3, horizon=2)

    assert len(dataset) == 8
    history, target = dataset[0]
    assert history.dtype == target.dtype == torch.float32
    torch.testing.assert_close(history, torch.tensor([10., 11., 12.]))
    torch.testing.assert_close(target, torch.tensor([13., 14.]))

    history, target = dataset[len(dataset) - 1]
    torch.testing.assert_close(history, torch.tensor([103., 104., 105.]))
    torch.testing.assert_close(target, torch.tensor([106., 107.]))


def test_multivariate_columns_price_fill_and_matching_windows(sales_frame):
    feature_columns = ["lag_1", "sell_price", "units_sold", "is_weekend"]
    multi = create_dataset(
        sales_frame,
        lookback=3,
        horizon=2,
        feature_columns=feature_columns,
        warm_up=1,
    )
    single = create_dataset(sales_frame, lookback=3, horizon=2, warm_up=1)

    assert len(multi) == 6
    assert multi.sample_index == single.sample_index
    assert multi.sample_index[0] == (0, 4)

    history, target = multi[0]
    assert history.shape == (3, 4)
    assert target.shape == (2,)
    assert history.dtype == target.dtype == torch.float32
    torch.testing.assert_close(history, torch.tensor([
        [10., 0., 11., 0.],
        [11., 0., 12., 1.],
        [12., 2.5, 13., 0.],
    ]))
    torch.testing.assert_close(target, torch.tensor([14., 15.]))

    for index in range(len(multi)):
        multi_history, multi_target = multi[index]
        single_history, single_target = single[index]
        assert torch.isfinite(multi_history).all()
        torch.testing.assert_close(multi_history[:, 2], single_history)
        torch.testing.assert_close(multi_target, single_target)

    assert sales_frame["sell_price"].null_count() == 6
    assert multi.series[0]["sell_price"].null_count() == 3


def test_future_changes_do_not_change_history(sales_frame):
    feature_columns = ["units_sold", "sell_price", "lag_1"]
    original = create_dataset(
        sales_frame, lookback=3, horizon=2,
        feature_columns=feature_columns, warm_up=1,
    )
    is_future = pl.col("date") >= date(2020, 1, 5)
    changed_frame = sales_frame.with_columns(
        pl.when(is_future).then(pl.col("units_sold") + 1000)
        .otherwise(pl.col("units_sold")).alias("units_sold"),
        pl.when(is_future).then(999.0)
        .otherwise(pl.col("sell_price")).alias("sell_price"),
        pl.when(is_future).then(999)
        .otherwise(pl.col("lag_1")).alias("lag_1"),
    )
    changed = create_dataset(
        changed_frame, lookback=3, horizon=2,
        feature_columns=feature_columns, warm_up=1,
    )

    history, target = original[0]
    changed_history, changed_target = changed[0]
    torch.testing.assert_close(changed_history, history)
    torch.testing.assert_close(changed_target, target + 1000)


@pytest.mark.parametrize("rows, expected_windows", [(5, 0), (6, 1), (7, 2)])
def test_warmup_window_boundaries(rows, expected_windows):
    sales = pl.DataFrame({
        "store_id": ["STORE_1"] * rows,
        "item_id": ["A"] * rows,
        "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(rows)],
        "units_sold": list(range(rows)),
        "lag_2": pl.Series([None, None] + list(range(rows - 2)), dtype=pl.Int64),
    })
    dataset = ForecastDataset(
        sales, lookback=2, horizon=2,
        feature_columns=["units_sold", "lag_2"], warmup=2,
    )

    assert len(dataset) == expected_windows
    if expected_windows:
        assert dataset.sample_index[0] == (0, 4)
        history, target = dataset[0]
        torch.testing.assert_close(history, torch.tensor([[2., 0.], [3., 1.]]))
        torch.testing.assert_close(target, torch.tensor([4., 5.]))
        assert torch.isfinite(history).all()
        _, final_target = dataset[len(dataset) - 1]
        assert final_target[-1].item() == rows - 1
