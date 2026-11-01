import polars as pl
from scripts.inspect_m5 import wide_to_long,convert_sales_date
import pytest
@pytest.mark.parametrize(
    "item_id, day, expected",
    [
        ("A", "d_1", 0),
        ("A", "d_2", 3),
        ("B", "d_1", 5),
        ("B", "d_2", 0),
    ],
)
def test_sales_mapping( item_id, day, expected):
    sales = pl.DataFrame({
        "id": ["A_STORE_1", "B_STORE_1"],
        "item_id": ["A", "B"],
        "dept_id": ["DEPT_1", "DEPT_1"],
        "cat_id": ["CAT_1", "CAT_1"],
        "store_id": ["STORE_1", "STORE_1"],
        "state_id": ["CA", "CA"],
        "d_1": [0, 5],
        "d_2": [3, 0],
    })

    result = wide_to_long(sales)

    actual = result.filter(
        (pl.col("item_id") == item_id)
        & (pl.col("d") == day)
    )["units_sold"].item()

    assert actual == expected
@pytest.mark.parametrize(
    "date_value, expected_exception,units_sold",
    [
        (20261027, TypeError,[3]),
    ],
)
def test_convert_sales_date_invalid_type(
    date_value, expected_exception, units_sold
):
    sales = pl.DataFrame({
        "date": [date_value],
        "units_sold": units_sold,
    })

    with pytest.raises(expected_exception):
        convert_sales_date(sales)

@pytest.mark.parametrize(
    "units_sold, expected_total",
    [
        ([3, 10], 13),
        ([0, 0], 0),
        ([3, None], 3),
    ],
)
def test_convert_sales_date_preserves_total(
    units_sold, expected_total
):
    sales = pl.DataFrame({
        "date": ["2026-10-27", "2026-10-28"],
        "units_sold": units_sold,
    })

    result = convert_sales_date(sales)

    assert result.schema["date"] == pl.Date
    assert result["units_sold"].sum() == expected_total
    assert result["units_sold"].sum() == sales["units_sold"].sum()