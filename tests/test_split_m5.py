import  polars as pl
from datetime import date,timedelta
from scripts.split_m5 import split_dataset_by_time
from scripts.train import create_dataset
def test_split_by_time_boundaries():
    sales = pl.DataFrame({
        "store_id": ["STORE_1"] * 7,
        "item_id": ["A"] * 7,
        "date": [
            date(2016, 3, 21) + timedelta(days=i)
            for i in range(7)
        ],
        "units_sold": [10, 11, 12, 13, 14, 15, 16],
    })

    dataset = create_dataset(sales, lookback=1, horizon=2)

    train, valid, test = split_dataset_by_time(
        dataset,
        train_end=date(2016, 3, 23),
        validation_end=date(2016, 3, 25),
        test_end=date(2016, 3, 27),
    )
    assert train.indices == [0]
    history, target = train[0]
    assert history.tolist() == [10.0]
    assert target.tolist() == [11.0, 12.0]
    assert valid.indices == [2]
    history, target = valid[0]
    assert history.tolist() == [12.0]
    assert target.tolist() == [13.0, 14.0]
    assert test.indices == [4]
    history, target = test[0]
    assert history.tolist() == [14.0]
    assert target.tolist() == [15.0, 16.0]

