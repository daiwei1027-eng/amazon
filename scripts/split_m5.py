from datetime import date
import polars as pl
from pathlib import Path
import yaml
def split_by_time(
    sales: pl.DataFrame,
    train_end: date,
    validation_end: date,
    test_end: date,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if not train_end < validation_end < test_end:
        raise ValueError("Split boundaries must be strictly increasing")
    train_sales=sales.filter(pl.col("date")<=train_end)
    validation_sales = sales.filter(
        (pl.col("date") > train_end)
        & (pl.col("date") <= validation_end)
    )
    test_sales = sales.filter(
        (pl.col("date") > validation_end)
        & (pl.col("date") <= test_end)
    )
    return train_sales, validation_sales, test_sales
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_dir = project_root / config["data"]["directory"]
    output_config = config["data"]["output"]
    output_dir = data_dir / output_config["directory"]
    exp_sub_filename = output_config["exp_sub_filename"]
    exp_filename_path = output_dir / exp_sub_filename
    train_end = date.fromisoformat(config["split"]["train_end"])
    validation_end = date.fromisoformat(config["split"]["validation_end"])
    test_end = date.fromisoformat(config["split"]["test_end"])
    sales = pl.read_parquet(exp_filename_path)
    train_sales, validation_sales, test_sales = split_by_time(
        sales, train_end, validation_end, test_end
    )
    train_name=config["split"]["train"]
    validation_name=config["split"]["valid"]
    test_name=config["split"]["test"]
    train_path=output_dir / train_name
    validation_path=output_dir / validation_name
    test_path=output_dir / test_name
    train_sales.write_parquet(train_path)
    validation_sales.write_parquet(validation_path)
    test_sales.write_parquet(test_path)



if __name__ == "__main__":
    main()
