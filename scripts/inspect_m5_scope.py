from pathlib import Path
import polars as pl
import yaml
def sample_sales(
    sales: pl.DataFrame,
    store_ids: list[str],
    item_count: int,
    seed: int,
) -> pl.DataFrame:
    metadata = sales.select(
        ["store_id", "item_id", "cat_id"]
    ).unique()
    target_metadata = metadata.filter(
        pl.col("store_id").is_in(store_ids)
    )
    item_coverage = target_metadata.group_by("item_id").agg(
        pl.col("store_id").n_unique().alias("store_count")
    )
    common_items = item_coverage.filter(pl.col("store_count") == len(store_ids))
    sampled_item_ids = (
        common_items
        .select("item_id")
        .sort("item_id")
        .sample(n=item_count, seed=seed)
        ["item_id"]
        .to_list()
    )

    return sales.filter(
        (pl.col("store_id").is_in(store_ids))
        & (pl.col("item_id").is_in(sampled_item_ids))
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_dir = project_root / config["data"]["directory"]
    output_config = config["data"]["output"]
    parquet_path = data_dir / output_config["directory"] / output_config["filename"]
    output_dir = data_dir / output_config["directory"]
    exp_sub_filename = output_config["exp_sub_filename"]
    exp_filename_path = output_dir / exp_sub_filename
    store_ids=config["data"]["sampling"]["store_ids"]
    item_counts = config["data"]["sampling"]["item_counts"]
    seed = config["data"]["sampling"]["seed"]
    sales = pl.read_parquet(parquet_path)
    sampled_sales=sample_sales(sales, store_ids, item_counts, seed=seed)
    if sampled_sales["store_id"].n_unique() != len(store_ids):
        raise ValueError("Unexpected store count")
    if sampled_sales["item_id"].n_unique() != item_counts:
        raise ValueError("Unexpected item count")
    series_count = sampled_sales.select(
        ["store_id", "item_id"]
    ).unique().height

    if series_count != len(store_ids) * item_counts:
        raise ValueError("Incomplete store-item combinations")
    sampled_sales.write_parquet(exp_filename_path)
    reloader = pl.read_parquet(exp_filename_path)
    if reloader.shape != sampled_sales.shape:
        raise ValueError("Unexpected shape")
    if sampled_sales.schema != reloader.schema:
        raise TypeError("Unexpected schema")
    if sampled_sales["units_sold"].sum() != reloader["units_sold"].sum():
        raise ValueError("units_sold != reloaded")
    if sampled_sales["sell_price"].null_count() != reloader["sell_price"].null_count():
        raise ValueError("sell_price != reloaded")
    print(f"Parquet saved and reloaded: {exp_filename_path}")
if __name__ == "__main__":
    main()
