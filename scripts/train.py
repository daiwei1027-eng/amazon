import polars as pl
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from etc.losses import mae_loss
from models.baseline import NaiveBaseline
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from scripts.split_m5 import split_dataset_by_time
from etc.dataset import ForecastDataset
from datetime import date
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
    total_loss = 0.0
    total_values = 0
    for past_target, target in tqdm.tqdm(dataloader):
        past_target = past_target.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        prediction = model(past_target)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        batch_values = target.numel()
        total_loss += loss.item() * batch_values
        total_values += batch_values
    if total_values == 0:
        raise ValueError("No prediction targets available for training")
    return total_loss / total_values
def evaluate(model,device,dataloader,criterion):
    total_loss = 0.0
    total_values = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for past_target, target in tqdm.tqdm(dataloader):
            past_target = past_target.to(device)
            target = target.to(device)
            prediction = model(past_target)
            loss = criterion(prediction, target)
            batch_values = target.numel()
            total_loss += loss.item() * batch_values
            total_values += batch_values
        if total_values == 0:
            raise ValueError("No prediction targets available for validation")
        return total_loss / total_values
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
):
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if optimizer is None:
        valid_loss = evaluate(
            model, device, valid_dataloader, criterion
        )
        if writer is not None:
            writer.add_scalar("MAE/valid", valid_loss, global_step=0)

        return None, valid_loss

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
        if writer is not None:
            writer.add_scalar("MAE/train", train_loss, epoch + 1)
            writer.add_scalar("MAE/valid", valid_loss, epoch + 1)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_mae={train_loss:.6f} "
            f"valid_mae={valid_loss:.6f}"
        )

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
    criterion = mae_loss
    baseline_model = NaiveBaseline(horizon).to(device)
    run_name = f"naive_{datetime.now():%Y%m%d_%H%M%S_%f}"
    log_dir = project_root / "logs" / "tensorboard" / run_name
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        train_mae, valid_mae = fit(
            baseline_model,
            device,
            train_dataloader,
            valid_dataloader,
            criterion,
            epochs=epochs,
            optimizer=None,
            writer=writer,
        )

    print(f"Naive validation MAE: {valid_mae:.6f}")


if __name__ == "__main__":
    main()
