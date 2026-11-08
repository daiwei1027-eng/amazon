from datetime import date, timedelta

import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from etc.dataset import ForecastDataset
from etc.features import add_lag_features, add_rolling_features
from etc.tft_dataset import TFTDataset
from scripts.split_m5 import split_dataset_by_time


def make_sales_frame(days=8):
    rows = []
    for store_id, item_id, base, price in [
        ("STORE_2", "B", 200, 7.0),
        ("STORE_1", "B", 100, 5.0),
        ("STORE_1", "A", 10, 2.5),
    ]:
        for day in range(days):
            current_date = date(2015, 12, 29) + timedelta(days=day)
            wday = (current_date.weekday() + 2) % 7 + 1
            rows.append({
                "store_id": store_id,
                "item_id": item_id,
                "date": current_date,
                "units_sold": base + day,
                "sell_price": None if day < 3 else price,
                "lag_1": None if day == 0 else base + day - 1,
                "wday": wday,
                "is_weekend": wday in (1, 2),
                "month": current_date.month,
                "year_offset": current_date.year - 2011,
            })
    return pl.DataFrame(rows).reverse()


@pytest.fixture
def sales_frame():
    return make_sales_frame()


@pytest.fixture
def dataset_kwargs():
    return {
        "lookback": 3,
        "horizon": 2,
        "warmup": 1,
        "history_observed_features": ["lag_1", "sell_price", "units_sold"],
        "known_future_features": ["wday", "is_weekend", "month", "year_offset"],
        "static_mappings": {
            "store_id": {"STORE_1": 1, "STORE_2": 0},
            "item_id": {"A": 1, "B": 0},
        },
    }


def test_window_values_feature_order_dtypes_and_price_fill(sales_frame, dataset_kwargs):
    dataset = TFTDataset(sales_frame, **dataset_kwargs)
    inputs, target = dataset[0]

    assert set(inputs) == {"past_observed", "past_known", "future_known", "static"}
    assert inputs["past_observed"].shape == (3, 3)
    assert inputs["past_known"].shape == (3, 4)
    assert inputs["future_known"].shape == (2, 4)
    assert inputs["static"].shape == (2,)
    assert target.shape == (2,)
    for name in ("past_observed", "past_known", "future_known"):
        assert inputs[name].dtype == torch.float32
        assert torch.isfinite(inputs[name]).all()
    assert inputs["static"].dtype == torch.long
    assert target.dtype == torch.float32

    torch.testing.assert_close(inputs["past_observed"], torch.tensor([
        [10., 0., 11.],
        [11., 0., 12.],
        [12., 2.5, 13.],
    ]))
    torch.testing.assert_close(inputs["past_known"], torch.tensor([
        [5., 0., 12., 4.],
        [6., 0., 12., 4.],
        [7., 0., 1., 5.],
    ]))
    torch.testing.assert_close(inputs["future_known"], torch.tensor([
        [1., 1., 1., 5.],
        [2., 1., 1., 5.],
    ]))
    torch.testing.assert_close(inputs["static"], torch.tensor([1, 1]))
    torch.testing.assert_close(target, torch.tensor([14., 15.]))
    assert sales_frame["sell_price"].null_count() == 9
    assert dataset.series[0]["sell_price"].null_count() == 3


def test_series_are_isolated_and_use_supplied_static_ids(sales_frame, dataset_kwargs):
    dataset = TFTDataset(sales_frame, **dataset_kwargs)
    baseline = ForecastDataset(sales_frame, lookback=3, horizon=2, warmup=1)

    assert len(dataset) == 9
    assert dataset.sample_index == baseline.sample_index
    for index, history, target, static in [
        (0, [11., 12., 13.], [14., 15.], [1, 1]),
        (3, [101., 102., 103.], [104., 105.], [1, 0]),
        (6, [201., 202., 203.], [204., 205.], [0, 0]),
    ]:
        inputs, labels = dataset[index]
        torch.testing.assert_close(inputs["past_observed"][:, 2], torch.tensor(history))
        torch.testing.assert_close(labels, torch.tensor(target))
        torch.testing.assert_close(inputs["static"], torch.tensor(static))


def test_static_ids_follow_mapping_field_order(sales_frame, dataset_kwargs):
    mappings = dataset_kwargs["static_mappings"]
    reordered = {
        "item_id": mappings["item_id"],
        "store_id": mappings["store_id"],
    }
    original = TFTDataset(sales_frame, **dataset_kwargs)
    changed = TFTDataset(
        sales_frame, **{**dataset_kwargs, "static_mappings": reordered}
    )
    before, target_before = original[3]
    after, target_after = changed[3]

    torch.testing.assert_close(before["static"], torch.tensor([1, 0]))
    torch.testing.assert_close(after["static"], torch.tensor([0, 1]))
    for name in ("past_observed", "past_known", "future_known"):
        torch.testing.assert_close(after[name], before[name])
    torch.testing.assert_close(target_after, target_before)


def test_future_observations_change_labels_but_never_inputs(sales_frame, dataset_kwargs):
    is_future = pl.col("date") >= date(2016, 1, 2)
    changed_frame = sales_frame.with_columns(
        pl.when(is_future).then(pl.col("units_sold") + 1000)
        .otherwise(pl.col("units_sold")).alias("units_sold"),
        pl.when(is_future).then(999.0)
        .otherwise(pl.col("sell_price")).alias("sell_price"),
        pl.when(is_future).then(999)
        .otherwise(pl.col("lag_1")).alias("lag_1"),
    )
    original = TFTDataset(sales_frame, **dataset_kwargs)
    changed = TFTDataset(changed_frame, **dataset_kwargs)
    before, target_before = original[0]
    after, target_after = changed[0]

    for name in before:
        torch.testing.assert_close(after[name], before[name])
    torch.testing.assert_close(target_after, target_before + 1000)


def test_future_calendar_changes_only_future_known(sales_frame, dataset_kwargs):
    changed_frame = sales_frame.with_columns(
        pl.when(pl.col("date") >= date(2016, 1, 2)).then(7)
        .otherwise(pl.col("month")).alias("month")
    )
    before, target_before = TFTDataset(sales_frame, **dataset_kwargs)[0]
    after, target_after = TFTDataset(changed_frame, **dataset_kwargs)[0]

    for name in ("past_observed", "past_known", "static"):
        torch.testing.assert_close(after[name], before[name])
    torch.testing.assert_close(target_after, target_before)
    torch.testing.assert_close(after["future_known"][:, 2], torch.tensor([7., 7.]))
    torch.testing.assert_close(
        after["future_known"][:, [0, 1, 3]],
        before["future_known"][:, [0, 1, 3]],
    )


@pytest.mark.parametrize("rows, expected_windows", [(5, 0), (6, 1), (7, 2)])
def test_warmup_window_boundaries(sales_frame, dataset_kwargs, rows, expected_windows):
    short = sales_frame.filter(
        (pl.col("store_id") == "STORE_1") & (pl.col("item_id") == "A")
    ).sort("date").head(rows)
    kwargs = {**dataset_kwargs, "lookback": 2, "warmup": 2}
    dataset = TFTDataset(short, **kwargs)

    assert len(dataset) == expected_windows
    if expected_windows:
        assert dataset.sample_index[0] == (0, 4)
        inputs, target = dataset[0]
        torch.testing.assert_close(inputs["past_observed"][:, 2], torch.tensor([12., 13.]))
        torch.testing.assert_close(target, torch.tensor([14., 15.]))


def test_full_feature_windows_batch_with_dataloader(dataset_kwargs):
    sales = add_lag_features(make_sales_frame(days=114), lags=[1, 7, 28])
    sales = add_rolling_features(sales, windows=[7, 14, 28])
    kwargs = {
        **dataset_kwargs,
        "lookback": 56,
        "horizon": 30,
        "warmup": 28,
        "history_observed_features": [
            "units_sold", "sell_price", "lag_1", "lag_7", "lag_28",
            "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
            "rolling_mean_28", "rolling_std_28",
        ],
    }
    dataset = TFTDataset(sales, **kwargs)
    inputs, target = next(iter(DataLoader(dataset, batch_size=3, shuffle=False)))

    assert len(dataset) == 3
    assert inputs["past_observed"].shape == (3, 56, 11)
    assert inputs["past_known"].shape == (3, 56, 4)
    assert inputs["future_known"].shape == (3, 30, 4)
    assert inputs["static"].shape == (3, 2)
    assert inputs["static"].dtype == torch.long
    assert target.shape == (3, 30)
    for name in ("past_observed", "past_known", "future_known"):
        assert inputs[name].dtype == torch.float32
        assert torch.isfinite(inputs[name]).all()
    assert target.dtype == torch.float32
    assert torch.isfinite(target).all()
    torch.testing.assert_close(inputs["static"], torch.tensor([[1, 1], [1, 0], [0, 0]]))
    torch.testing.assert_close(inputs["past_observed"][:, -1, 0], torch.tensor([93., 183., 283.]))
    torch.testing.assert_close(target[:, 0], torch.tensor([94., 184., 284.]))
    torch.testing.assert_close(target[:, -1], torch.tensor([123., 213., 313.]))


def test_existing_time_split_accepts_tft_dataset(sales_frame, dataset_kwargs):
    sales = sales_frame.filter(
        (pl.col("store_id") == "STORE_1") & (pl.col("item_id") == "A")
    )
    dataset = TFTDataset(sales, **{**dataset_kwargs, "lookback": 1})
    train, valid, test = split_dataset_by_time(
        dataset,
        train_end=date(2016, 1, 1),
        validation_end=date(2016, 1, 3),
        test_end=date(2016, 1, 5),
    )

    assert train.indices == [0]
    assert valid.indices == [2]
    assert test.indices == [4]
    for subset, expected_target in [
        (train, [12., 13.]), (valid, [14., 15.]), (test, [16., 17.]),
    ]:
        inputs, target = subset[0]
        assert inputs["past_observed"].shape == (1, 3)
        torch.testing.assert_close(target, torch.tensor(expected_target))
