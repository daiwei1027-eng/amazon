import polars as pl
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from etc.losses import mae_loss,rmse_loss,wape_loss
from models.baseline import NaiveBaseline
from models.SeasonNaive import SeasonalNaiveBaseline
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from scripts.split_m5 import split_dataset_by_time
from etc.dataset import ForecastDataset
from datetime import date
from models.lstm import LSTM
def create_dataset(
    sales: pl.DataFrame,lookback: int,horizon: int,
)->Dataset:
    for name, value in [
        ("lookback", lookback),
        ("horizon", horizon),
    ]:
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    return ForecastDataset(sales,lookback,horizon)
def create_dataloader(dataset: Dataset, batch_size: int,shuffle:bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

def train_one_epoch(
    model,
    device,
    dataloader,
    criterion,
    *,
    optimizer,
):
    model.to(device)
    model.train()
    total_error = 0.0
    total_weight = 0.0
    for past_target, target in tqdm.tqdm(dataloader):
        past_target = past_target.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        prediction = model(past_target)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        error = prediction.detach() - target.detach()

        if criterion is mae_loss:
            total_error += error.abs().sum().item()
            total_weight += target.numel()

        elif criterion is rmse_loss:
            total_error += error.square().sum().item()
            total_weight += target.numel()

        elif criterion is wape_loss:
            total_error += error.abs().sum().item()
            total_weight += target.detach().abs().sum().item()

        else:
            raise ValueError("Unsupported criterion")
    if total_weight == 0:
        raise ValueError("No targets available or metric denominator is zero")

    result = total_error / total_weight
    if criterion is rmse_loss:
        result = result ** 0.5

    return result

def evaluate(model,device,dataloader,criterion):
    total_error = 0.0
    total_weight = 0.0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for past_target, target in tqdm.tqdm(dataloader):
            past_target = past_target.to(device)
            target = target.to(device)
            prediction = model(past_target)
            if prediction.shape != target.shape:
                raise ValueError("Shape of prediction and target do not match")

            error = prediction.detach() - target.detach()

            if criterion is mae_loss:
                total_error += error.abs().sum().item()
                total_weight += target.numel()

            elif criterion is rmse_loss:
                total_error += error.square().sum().item()
                total_weight += target.numel()

            elif criterion is wape_loss:
                total_error += error.abs().sum().item()
                total_weight += target.detach().abs().sum().item()

            else:
                raise ValueError("Unsupported criterion")

        if total_weight == 0:
            raise ValueError("No targets available or metric denominator is zero")

        result = total_error / total_weight
        if criterion is rmse_loss:
            result = result ** 0.5

        return result
def fit(
    model,
    device,
    train_dataloader,
    valid_dataloader,
    criterion,
    epochs: int,
    *,
    optimizer=None,

writer: SummaryWriter | None = None,
        patience: int|None = None,
        checkpoint_path:Path | None = None,
        checkpoint_interval:int|None = None,
):
    metric_name = criterion.__name__.removesuffix("_loss").upper()
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be positive")

    if checkpoint_interval is not None:
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required for periodic checkpoints"
            )

    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    if optimizer is None:
        valid_loss = evaluate(
            model, device, valid_dataloader, criterion
        )
        if writer is not None:
            writer.add_scalar(
                f"{metric_name}/valid", valid_loss, global_step=0
            )

        return None, valid_loss
    best_loss = float("inf")
    best_epoch = 0
    best_train_loss = None
    bad_epochs = 0
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            device,
            train_dataloader,
            criterion,
            optimizer=optimizer,
        )
        valid_loss = evaluate(
            model, device, valid_dataloader, criterion
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_train_loss = train_loss
            bad_epochs = 0

            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": best_train_loss,
                        "valid_loss": best_loss,
                        "metric_name": metric_name,
                    },
                    checkpoint_path / "best.pt",
                )
        else:
            bad_epochs += 1
        if writer is not None:
            writer.add_scalar(
                f"{metric_name}/train", train_loss, epoch + 1
            )
            writer.add_scalar(
                f"{metric_name}/valid", valid_loss, epoch + 1
            )
        if (
                checkpoint_interval is not None
                and (epoch + 1) % checkpoint_interval == 0
        ):
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "metric_name": metric_name,

            }
            save_index = (epoch + 1) // checkpoint_interval
            checkpoint_file = (
                    checkpoint_path / f"checkpoint_{save_index:04d}.pt"
            )
            torch.save(checkpoint, checkpoint_file)
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_{metric_name}={train_loss:.6f} "
            f"valid_{metric_name}={valid_loss:.6f}"
        )
        if patience is not None and bad_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best epoch={best_epoch}, "
                f"best_valid_loss={best_loss:.6f}"
            )
            break
    return train_loss, valid_loss

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_config = config["data"]
    output_config = data_config["output"]
    data_dict = project_root / config["data"]["directory"] / config["features"]["directory"]/config["features"]["filename"]
    sales = pl.read_parquet(data_dict)
    lookback = config["train"]["lookback"]
    horizon = config["train"]["horizon"]
    train_end = date.fromisoformat(config["split"]["train_end"])
    validation_end = date.fromisoformat(
        config["split"]["validation_end"]
    )
    test_end = date.fromisoformat(config["split"]["test_end"])
    sales_dataset=create_dataset(sales,lookback,horizon)
    train_dataset,valid_dataset,test_dataset = split_dataset_by_time(sales_dataset,train_end,validation_end,test_end)
    train_batch_size=config["train"]["train_batch_size"]
    valid_batch_size=config["train"]["valid_batch_size"]
    epochs = config["train"]["epochs"]
    train_dataloader = (create_dataloader(train_dataset, train_batch_size, shuffle=True))
    valid_dataloader = (create_dataloader(valid_dataset, valid_batch_size, shuffle=False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_functions = {
        "mae": mae_loss,
        "rmse": rmse_loss,
        "wape": wape_loss,
    }

    criterion = loss_functions[config["train"]["criterion"]]
    season_length=config["train"]["season_length"]
    lstm_num_layers=config["train"]["lstm"]["num_layers"]
    lstm_hidden_size=config["train"]["lstm"]["hidden_size"]
    models = {
        "naive": NaiveBaseline(horizon),
        "seasonal_naive": SeasonalNaiveBaseline(horizon, season_length),
        "lstm":LSTM(horizon=horizon,hidden_size=lstm_hidden_size,num_layers=lstm_num_layers),
    }

    model = models[config["train"]["model"]].to(device)
    model_name = model.__class__.__name__
    run_name = f"{model_name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    log_dir = project_root / "logs" / "tensorboard" / run_name
    checkpoint_path = (
            project_root/data_config["directory"]/ run_name
            / "checkpoints"
    )
    checkpoint_interval = config["train"]["checkpoint_interval"]
    patience = config["train"]["patience"]
    lr = config["train"]["optimizer"]["lr"]
    weight_decay = config["train"]["optimizer"]["weight_decay"]
    optimizer=None
    if config["train"]["model"]=="lstm":
        optimizer=torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        train_loss, valid_loss = fit(
            model,
            device,
            train_dataloader,
            valid_dataloader,
            criterion,
            epochs=epochs,
            optimizer=optimizer,
            writer=writer,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            patience=patience,
        )

    print(f": Valid Loss{valid_loss:.6f}")


if __name__ == "__main__":
    main()
