import polars as pl
from scripts.inspect_m5 import wide_to_long
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