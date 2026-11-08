from scripts.evaluate import evaluate
from scripts.create_dataset import create_dataset
from scripts.create_dataloader import create_dataloader
from etc.losses import mae_loss,rmse_loss,wape_loss
from scripts.split_m5 import split_dataset_by_time
from models.lstm import LSTM
import torch
import polars as pl
from models.baseline import NaiveBaseline
from models.SeasonNaive import SeasonalNaiveBaseline
from pathlib import Path
import yaml
from datetime import date
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        r"C:\Users\admin\PycharmProjects\amazon\data\LSTM_20261108_092617_264492\checkpoints\best.pt",
        map_location=device,
        weights_only=True,
    )
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_dict = project_root / config["data"]["directory"] / config["features"]["directory"] / config["features"][
        "filename"]
    sales = pl.read_parquet(data_dict)
    lookback = config["train"]["lookback"]
    horizon = config["train"]["horizon"]
    train_end = date.fromisoformat(config["split"]["train_end"])
    validation_end = date.fromisoformat(
        config["split"]["validation_end"]
    )
    test_end = date.fromisoformat(config["split"]["test_end"])
    feature_columns = config["train"]["lstm"]["feature_columns"]
    input_size = len(feature_columns) if feature_columns is not None else 1
    warm_up = config["features"]["warm_up"]
    sales_dataset = create_dataset(
        sales,
        lookback,
        horizon,
        feature_columns=feature_columns,
        warm_up=warm_up,
    )
    train_dataset, valid_dataset, test_dataset = split_dataset_by_time(sales_dataset, train_end, validation_end,
                                                                       test_end)
    valid_batch_size = config["train"]["valid_batch_size"]
    valid_dataloader = (create_dataloader(valid_dataset, valid_batch_size, shuffle=False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_functions = {
        "mae": mae_loss,
        "rmse": rmse_loss,
        "wape": wape_loss,
    }


    criterion = loss_functions[config["train"]["criterion"]]
    lstm_config = config["train"]["lstm"]

    model = LSTM(
        horizon=horizon,
        hidden_size=lstm_config["hidden_size"],
        num_layers=lstm_config["num_layers"],
        input_size=input_size,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    results = {}

    for name, criterion in loss_functions.items():
        results[name.upper()] = evaluate(
            model, device, valid_dataloader, criterion
        )

    for name, value in results.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()