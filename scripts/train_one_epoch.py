import tqdm
from etc.losses import mae_loss, rmse_loss, wape_loss
import mlflow
def train_one_epoch(
    model,
    device,
    dataloader,
    criterion,
    *,
    optimizer,
    logs_enable: bool,
    epoch_num: int|None = None,
):
    model.to(device)
    model.train()
    total_error = 0.0
    total_weight = 0.0
    for past_target, target in tqdm.tqdm(dataloader):
        if isinstance(past_target, dict):
            past_target = {
                name: tensor.to(device)
                for name, tensor in past_target.items()
            }
        else:
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

    if logs_enable:
        if epoch_num is None:
            raise ValueError("No epoch provided")
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run")
        metric_name = criterion.__name__.removesuffix("_loss").lower()
        mlflow.log_metric(
            key=f"train_{metric_name}",
            value=result,
            step=epoch_num,
        )
    return result
