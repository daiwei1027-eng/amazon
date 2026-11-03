import torch

def mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Shape of prediction and target do not match")
    return torch.mean(torch.abs(prediction - target))