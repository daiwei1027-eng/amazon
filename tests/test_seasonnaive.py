from models.SeasonNaive import SeasonalNaiveBaseline
import torch
def test_seasonnaive():
    model = SeasonalNaiveBaseline(horizon=30, season_length=7)
    x = torch.tensor([
        [99., 98., 1., 2., 3., 4., 5., 6., 7.],
        [99., 98., 10., 20., 30., 40., 50., 60., 70.],
    ])
    prediction = model(x)
    last_week = x[:, -7:]
    assert prediction.shape == (2, 30)
    assert torch.equal(prediction[:, :7], last_week )
    assert torch.equal(prediction[:, 7:14],last_week)
    assert torch.equal(prediction[:, 28:], last_week[:, :2])