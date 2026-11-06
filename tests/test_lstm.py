import pytest
import torch

from models.lstm import LSTM


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize(
    "input_size, input_shape",
    [(None, (2, 56)), (1, (2, 56, 1)), (12, (2, 56, 12))],
    ids=["single_default", "single_3d", "multivariate_12"],
)
def test_lstm_output_and_gradients(input_size, input_shape, num_layers):
    kwargs = {"horizon": 30, "hidden_size": 8, "num_layers": num_layers}
    if input_size is not None:
        kwargs["input_size"] = input_size
    model = LSTM(**kwargs)
    x = torch.rand(input_shape)
    target = torch.rand(2, 30)

    prediction = model(x)
    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()

    loss = (prediction - target).abs().mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_lstm_batch_samples_are_independent():
    model = LSTM(horizon=3, hidden_size=8, num_layers=2, input_size=12)
    model.eval()
    x = torch.rand(3, 6, 12)

    with torch.no_grad():
        batched = model(x)
        separate = torch.cat([model(sample.unsqueeze(0)) for sample in x], dim=0)

    torch.testing.assert_close(batched, separate, rtol=1e-5, atol=1e-6)
