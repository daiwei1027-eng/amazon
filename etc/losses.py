import torch

def mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Shape of prediction and target do not match")
    return torch.mean(torch.abs(prediction - target))
def rmse_loss(prediction, target) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Shape of prediction and target do not match")
    return torch.sqrt(torch.mean(torch.pow(prediction - target, 2)))
def wape_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Shape of prediction and target do not match")

    denominator = target.abs().sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined when total target is zero")

    return (prediction - target).abs().sum() / denominator