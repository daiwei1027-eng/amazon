import  polars as pl
from datetime import date
from scripts.split_m5 import split_by_time
def test_split_by_time_boundaries():
    sales = pl.DataFrame({
        "date": [
            date(2016, 3, 22),
            date(2016, 3, 23),
            date(2016, 3, 24),
            date(2016, 4, 22),
            date(2016, 4, 23),
            date(2016, 5, 22),
        ],
    })
    train_end=date.fromisoformat("2016-03-23")
    valid_end = date.fromisoformat("2016-04-22")
    test_end = date.fromisoformat("2016-05-22")
    train,valid,test = split_by_time(sales,train_end,valid_end,test_end)
    assert train["date"].to_list() == [
        date(2016, 3, 22),
        date(2016, 3, 23),
    ]
    assert valid["date"].to_list() == [
        date(2016, 3, 24),
        date(2016, 4, 22)
    ]
    assert test["date"].to_list() == [
        date(2016, 4, 23),
        date(2016, 5, 22),
    ]

