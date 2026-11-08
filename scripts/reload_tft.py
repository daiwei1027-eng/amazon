from datetime import date
from pathlib import Path

import polars as pl
import torch
import yaml

from etc.losses import mae_loss, rmse_loss, wape_loss
from etc.tft_dataset import TFTDataset
from models.tft import TFT
from scripts.create_dataloader import create_dataloader
from scripts.evaluate import evaluate
from scripts.split_m5 import split_dataset_by_time


RUN_NAME = "TFT_20261108_133756_506473"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = (
        project_root / config["data"]["directory"]
        / RUN_NAME / "checkpoints" / "best.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    data_path = (
        project_root / config["data"]["directory"]
        / config["features"]["directory"] / config["features"]["filename"]
    )
    sales = pl.read_parquet(data_path)
    tft_config = config["train"]["tft"]
    observed_features = tft_config["history_observed_features"]
    known_features = tft_config["known_future_features"]

    # Match the static ID assignment used by train.py.
    static_mappings = {}
    for name in tft_config["static_features"]:
        values = sorted(sales[name].unique().to_list())
        static_mappings[name] = {
            value: index
            for index, value in enumerate(values)
        }
    static_cardinalities = {
        name: len(mapping)
        for name, mapping in static_mappings.items()
    }

    lookback = config["train"]["lookback"]
    horizon = config["train"]["horizon"]
    dataset = TFTDataset(
        sales,
        lookback,
        horizon,
        history_observed_features=observed_features,
        known_future_features=known_features,
        static_mappings=static_mappings,
        warmup=config["features"]["warm_up"],
    )
    _, valid_dataset, _ = split_dataset_by_time(
        dataset,
        train_end=date.fromisoformat(config["split"]["train_end"]),
        validation_end=date.fromisoformat(config["split"]["validation_end"]),
        test_end=date.fromisoformat(config["split"]["test_end"]),
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        batch_size=config["train"]["valid_batch_size"],
        shuffle=False,
    )

    model = TFT(
        horizon=horizon,
        num_observed_features=len(observed_features),
        num_known_features=len(known_features),
        static_cardinalities=static_cardinalities,
        hidden_size=tft_config["hidden_size"],
        num_layers=tft_config["num_layers"],
        dropout=tft_config["dropout"],
        num_heads=tft_config["num_heads"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss_functions = {
        "MAE": mae_loss,
        "RMSE": rmse_loss,
        "WAPE": wape_loss,
    }
    results = {
        name: evaluate(model, device, valid_dataloader, criterion)
        for name, criterion in loss_functions.items()
    }

    print("Best epoch:", checkpoint["epoch"])
    print(f"Saved {checkpoint['metric_name']}:", checkpoint["valid_loss"])
    for name, value in results.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
