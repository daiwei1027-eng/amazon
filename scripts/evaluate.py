import torch
import tqdm

from etc.losses import mae_loss, rmse_loss, wape_loss


def evaluate(model,device,dataloader,criterion):
    total_error = 0.0
    total_weight = 0.0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for past_target, target in tqdm.tqdm(dataloader):
            if isinstance(past_target, dict):
                past_target = {
                    name: tensor.to(device)
                    for name, tensor in past_target.items()
                }
            else:
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
