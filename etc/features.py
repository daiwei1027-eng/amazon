import polars as pl

def add_calendar_features(
    sales: pl.DataFrame,
) -> pl.DataFrame:
    return sales.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("wday").is_in([1, 2]).alias("is_weekend"),
    )
def add_lag_features(
    sales: pl.DataFrame,
    lags: list[int],
) -> pl.DataFrame:
    sorted_sales = sales.sort(
        ["store_id", "item_id", "date"]
    )
    expressions = []

    for lag in lags:
        expressions.append(
            pl.col("units_sold")
            .shift(lag)
            .over(["store_id", "item_id"])
            .alias(f"lag_{lag}")
        )

    return sorted_sales.with_columns(expressions)
