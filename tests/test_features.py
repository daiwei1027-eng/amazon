from etc.features import add_calendar_features,add_lag_features
import polars as pl
from datetime import date
def test_add_calendar_features():
    sales = pl.DataFrame({
        "date": [
            date(2022, 12, 31),  # 周六
            date(2023, 1, 1),  # 周日
            date(2023, 1, 2),  # 周一
        ],
        "wday": [1, 2, 3],
    })
    processed_sales = add_calendar_features(sales)
    assert processed_sales["year"].to_list() == [2022, 2023, 2023]
    assert processed_sales["month"].to_list() == [12, 1, 1]
    assert processed_sales["is_weekend"].to_list() == [True,True,False]
def test_lag_features():
    sales = pl.DataFrame({
        "store_id": ["STORE_1"] * 6,
        "item_id": ["A"] * 3 + ["B"] * 3,
        "date": [
                    date(2026, 10, 27),
                    date(2026, 10, 28),
                    date(2026, 10, 29),
                ] * 2,
        "units_sold": [2, 0, 5, 10, 20, 30],
    })

    result = add_lag_features(sales, lags=[1, 2])
    assert result["lag_1"].to_list() == [
        None, 2, 0,
        None, 10, 20,
    ]
    assert result["lag_2"].to_list() == [
        None, None, 2,  # A
        None, None, 10,  # B
    ]