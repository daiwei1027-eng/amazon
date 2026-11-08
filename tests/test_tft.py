from datetime import date, timedelta
import io

import polars as pl
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from etc.losses import mae_loss
from etc.tft_dataset import TFTDataset
from models.tft import TFT


def make_model(**overrides):
    options = {
        "horizon": 3,
        "num_observed_features": 3,
        "num_known_features": 2,
        "static_cardinalities": {"store_id": 2, "item_id": 5},
        "hidden_size": 8,
    }
    options.update(overrides)
    return TFT(**options)


def make_inputs(
    *, batch_size=2, lookback=7, horizon=3,
    num_observed_features=3, num_known_features=2, requires_grad=False,
):
    ids = torch.arange(batch_size)
    return {
        "past_observed": torch.randn(
            batch_size, lookback, num_observed_features, requires_grad=requires_grad
        ),
        "past_known": torch.randn(
            batch_size, lookback, num_known_features, requires_grad=requires_grad
        ),
        "future_known": torch.randn(
            batch_size, horizon, num_known_features, requires_grad=requires_grad
        ),
        "static": torch.stack([ids % 2, (ids + 1) % 5], dim=1),
    }


@pytest.mark.parametrize(
    "options, batch_size, lookback",
    [
        ({}, 2, 7),
        ({"num_layers": 2, "num_heads": 4, "dropout": 0.25}, 2, 7),
        ({"hidden_size": 64, "horizon": 30,
          "num_observed_features": 11, "num_known_features": 4}, 2, 56),
        ({"horizon": 1, "num_known_features": 1}, 1, 5),
    ],
    ids=["single_layer", "multiple_layers_and_heads", "full_dimensions", "single_sample_step"],
)
def test_predictions_shapes_gradients_and_unchanged_inputs(options, batch_size, lookback):
    torch.manual_seed(42)
    model = make_model(**options)
    horizon = options.get("horizon", 3)
    inputs = make_inputs(
        batch_size=batch_size,
        lookback=lookback,
        horizon=horizon,
        num_observed_features=options.get("num_observed_features", 3),
        num_known_features=options.get("num_known_features", 2),
        requires_grad=True,
    )
    original = {name: value.detach().clone() for name, value in inputs.items()}

    prediction = model(inputs)

    assert prediction.shape == (batch_size, horizon)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    for name in inputs:
        torch.testing.assert_close(inputs[name].detach(), original[name], rtol=0, atol=0)

    loss = mae_loss(prediction, torch.rand_like(prediction))
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for name in ("past_observed", "past_known", "future_known"):
        assert inputs[name].grad is not None, name
        assert torch.isfinite(inputs[name].grad).all(), name


@pytest.mark.parametrize("num_heads", [1, 4])
def test_attention_weights_are_normalized_and_causal(num_heads):
    model = make_model(num_heads=num_heads)
    model.eval()
    captured = {}

    def capture_attention(_layer, _args, output):
        captured["weights"] = output[1]

    handle = model.attention.register_forward_hook(capture_attention)
    try:
        with torch.no_grad():
            model(make_inputs())
    finally:
        handle.remove()

    weights = captured["weights"]
    assert weights.shape == (2, 10, 10)
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 10))
    assert torch.count_nonzero(weights.triu(1)) == 0


@pytest.mark.parametrize("cutoff", [1, 3])
def test_later_calendar_values_cannot_change_earlier_forecasts(cutoff):
    torch.manual_seed(42)
    model = make_model(horizon=5, num_heads=4)
    model.eval()
    inputs = make_inputs(horizon=5)
    changed = {name: value.clone() for name, value in inputs.items()}
    changed["future_known"][:, cutoff:] += 100

    with torch.no_grad():
        before = model(inputs)
        after = model(changed)

    torch.testing.assert_close(after[:, :cutoff], before[:, :cutoff])
    assert not torch.allclose(after[:, cutoff:], before[:, cutoff:])


def test_batch_samples_are_independent():
    torch.manual_seed(7)
    model = make_model(num_layers=2, num_heads=4)
    model.eval()
    inputs = make_inputs(batch_size=3)

    with torch.no_grad():
        together = model(inputs)
        separate = torch.cat([
            model({name: value[i:i + 1] for name, value in inputs.items()})
            for i in range(3)
        ], dim=0)

    torch.testing.assert_close(together, separate, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("future_length", [2, 4], ids=["too_short", "too_long"])
def test_future_length_must_match_horizon(future_length):
    model = make_model(horizon=3)

    with pytest.raises(ValueError, match="Future input length must equal horizon"):
        model(make_inputs(horizon=future_length))


@pytest.mark.parametrize("num_layers", [1, 2])
def test_dropout_configuration_and_eval_repeatability(num_layers):
    torch.manual_seed(11)
    model = make_model(num_layers=num_layers, dropout=0.25)
    dropout_layers = [layer for layer in model.modules() if isinstance(layer, nn.Dropout)]
    assert dropout_layers
    assert all(layer.p == 0.25 for layer in dropout_layers)
    expected_lstm_dropout = 0.25 if num_layers > 1 else 0.0
    assert model.encoder_lstm.dropout == expected_lstm_dropout
    assert model.decoder_lstm.dropout == expected_lstm_dropout
    inputs = make_inputs()

    model.train()
    with torch.no_grad():
        torch.manual_seed(20)
        first_train = model(inputs)
        torch.manual_seed(21)
        second_train = model(inputs)
    assert not torch.allclose(first_train, second_train)

    model.eval()
    with torch.no_grad():
        torch.manual_seed(20)
        first_eval = model(inputs)
        torch.manual_seed(21)
        second_eval = model(inputs)
    torch.testing.assert_close(first_eval, second_eval, rtol=0, atol=0)


def test_dataset_dataloader_and_adamw_step_work_together():
    torch.manual_seed(42)
    rows = []
    for store, item, base in [("STORE_2", "B", 100), ("STORE_1", "A", 10)]:
        for day in range(8):
            current_date = date(2016, 1, 29) + timedelta(days=day)
            wday = (current_date.weekday() + 2) % 7 + 1
            rows.append({
                "store_id": store,
                "item_id": item,
                "date": current_date,
                "units_sold": base + day,
                "sell_price": None if day < 3 else 2.5,
                "lag_1": None if day == 0 else base + day - 1,
                "wday": wday,
                "month": current_date.month,
                "is_weekend": wday in (1, 2),
                "year_offset": current_date.year - 2011,
            })
    mappings = {
        "store_id": {"STORE_1": 0, "STORE_2": 1},
        "item_id": {"A": 0, "B": 1},
    }
    dataset = TFTDataset(
        pl.DataFrame(rows).reverse(),
        lookback=3,
        horizon=2,
        warmup=1,
        history_observed_features=["units_sold", "sell_price", "lag_1"],
        known_future_features=["wday", "month", "is_weekend", "year_offset"],
        static_mappings=mappings,
    )
    inputs, target = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    model = make_model(
        horizon=2,
        num_known_features=4,
        static_cardinalities={name: len(mapping) for name, mapping in mappings.items()},
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before_bias = model.head.bias.detach().clone()
    optimizer.zero_grad()

    prediction = model(inputs)
    loss = mae_loss(prediction, target)
    loss.backward()
    optimizer.step()

    assert prediction.shape == target.shape == (2, 2)
    assert torch.isfinite(loss)
    assert not torch.equal(model.head.bias.detach(), before_bias)


def test_checkpoint_roundtrip_preserves_predictions():
    torch.manual_seed(42)
    options = {"num_layers": 2, "num_heads": 4, "dropout": 0.25}
    model = make_model(**options)
    model.eval()
    inputs = make_inputs()
    with torch.no_grad():
        expected = model(inputs)

    buffer = io.BytesIO()
    torch.save({"model_state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = make_model(**options)
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        actual = restored(inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
