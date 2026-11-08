import polars as pl
from pathlib import Path
import yaml
def add_calendar_features(
    sales: pl.DataFrame,
) -> pl.DataFrame:
    return sales.with_columns(
        pl.col("date").dt.year().alias("year"),
        (
                pl.col("date").dt.year() - 2011
        ).cast(pl.Float32).alias("year_offset"),
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
def add_rolling_features(
    sales: pl.DataFrame,
    windows: list[int],
) -> pl.DataFrame:
    sorted_sales = sales.sort(
        ["store_id", "item_id", "date"]
    )
    expressions = []
    for window in windows:
        expressions.append(
            pl.col("units_sold").shift(1).rolling_mean(
                window_size=window,
                min_samples=window,
            ).over(
                ["store_id", "item_id"]
            ).alias(f"rolling_mean_{window}")
        )
        expressions.append(
            pl.col("units_sold").shift(1).rolling_std(
    window_size=window,
    min_samples=window,
    ddof=1,
).over(
                ["store_id", "item_id"]
            ).alias(f"rolling_std_{window}")
        )
    return sorted_sales.with_columns(expressions)
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_dict=project_root / config["data"]["directory"] / config["features"]["directory"]/config["features"]["filename"]
    source_dict=project_root / config["data"]["directory"] / config["data"]["output"]["directory"]/config["data"]["output"]["exp_sub_filename"]
    train_sales=pl.read_parquet(source_dict)
    featured = add_calendar_features(train_sales)
    featured = add_lag_features(
        featured,
        lags=config["features"]["lags"],
    )
    featured = add_rolling_features(
        featured,
        windows=config["features"]["windows"],
    )
    data_dict.parent.mkdir(parents=True, exist_ok=True)
    featured.write_parquet(data_dict)
    print(featured.shape)
    print(f"Saved: {data_dict}")

if __name__ == "__main__":
    main()
