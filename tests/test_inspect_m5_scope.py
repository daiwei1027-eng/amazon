import  polars as pl
from scripts.inspect_m5_scope import sample_sales
def test_sample_sales():
    sales = pl.DataFrame({
        "item_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 2,
        "store_id": [
            "STORE_1", "STORE_1", "STORE_2", "STORE_2",
            "STORE_1", "STORE_1", "STORE_2", "STORE_2",
            "STORE_1", "STORE_1",
        ],
        "cat_id": ["CAT_1"] * 10,
        "date": ["2026-10-27", "2026-10-28"] * 5,
        "units_sold": [0, 3, 5, 0, 1, 2, 0, 4, 6, 0],
    })
    sampled_sales = sample_sales(sales, store_ids=["STORE_1", "STORE_2"],item_count=1,seed=42)
    assert sampled_sales.height == 4

    assert sampled_sales["item_id"].n_unique() == 1
    assert set(sampled_sales["item_id"].to_list()) <= {"A", "B"}

    assert set(sampled_sales["store_id"].to_list()) == {
        "STORE_1", "STORE_2"
    }
    assert set(sampled_sales["date"].to_list()) == {
        "2026-10-27", "2026-10-28"
    }


