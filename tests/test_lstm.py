from models.lstm import LSTM
import torch
def test_lstm():
    model = LSTM(horizon=30, hidden_size=64, num_layers=1)
    x = torch.rand(2, 56)
    target = torch.rand(2, 30)

    prediction = model(x)
    assert prediction.shape == target.shape

    loss = (prediction - target).abs().mean()
    loss.backward()

    assert model.lstm.weight_ih_l0.grad is not None
    assert torch.isfinite(model.lstm.weight_ih_l0.grad).all()