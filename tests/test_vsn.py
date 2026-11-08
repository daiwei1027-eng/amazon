import pytest
import torch
import torch.nn as nn

from models.vsn import VariableSelectionNetwork


@pytest.mark.parametrize(
    "input_shape, context_shape",
    [
        ((3, 2, 8), None),
        ((2, 56, 15, 8), (2, 5)),
        ((2, 30, 4, 8), (2, 30, 5)),
    ],
    ids=["static", "history_with_static_context", "future_with_temporal_context"],
)
def test_selection_shapes_normalized_weights_and_gradients(input_shape, context_shape):
    torch.manual_seed(42)
    model = VariableSelectionNetwork(
        num_features=input_shape[-2],
        hidden_size=input_shape[-1],
        context_size=context_shape[-1] if context_shape is not None else None,
    )
    x = torch.randn(input_shape, requires_grad=True)
    context = (
        torch.randn(context_shape, requires_grad=True)
        if context_shape is not None
        else None
    )

    selected, weights = model(x, context)

    assert selected.shape == input_shape[:-2] + (input_shape[-1],)
    assert weights.shape == input_shape[:-1]
    assert torch.isfinite(selected).all()
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert (weights <= 1).all()
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(input_shape[:-2])
    )

    target = torch.randn_like(selected)
    (selected - target).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    if context is not None:
        assert context.grad is not None
        assert torch.isfinite(context.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


class FixedWeightLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("logits", torch.tensor([0.25, 0.75]).log())

    def forward(self, x, context=None):
        return self.logits.expand(*x.shape[:-1], 2)


@pytest.mark.parametrize("temporal", [False, True], ids=["static", "temporal"])
def test_known_weights_select_the_expected_values(temporal):
    model = VariableSelectionNetwork(num_features=2, hidden_size=2)
    model.weight_grn = FixedWeightLogits()
    model.feature_grns = nn.ModuleList([nn.Identity(), nn.Identity()])
    x = torch.tensor([
        [[1., 2.], [5., 6.]],
        [[10., 20.], [50., 60.]],
    ])
    expected = torch.tensor([[4., 5.], [40., 50.]])
    if temporal:
        x = torch.stack([x, 2 * x, 3 * x], dim=1)
        expected = torch.stack([expected, 2 * expected, 3 * expected], dim=1)

    selected, weights = model(x)

    torch.testing.assert_close(selected, expected)
    torch.testing.assert_close(weights[..., 0], torch.full(weights.shape[:-1], 0.25))
    torch.testing.assert_close(weights[..., 1], torch.full(weights.shape[:-1], 0.75))


def test_static_context_matches_explicit_broadcast_over_time():
    torch.manual_seed(7)
    model = VariableSelectionNetwork(num_features=3, hidden_size=8, context_size=5)
    model.eval()
    x = torch.randn(2, 6, 3, 8)
    context = torch.randn(2, 5)
    expanded_context = context.unsqueeze(1).expand(-1, 6, -1)

    with torch.no_grad():
        selected, weights = model(x, context)
        expanded_selected, expanded_weights = model(x, expanded_context)

    torch.testing.assert_close(selected, expanded_selected)
    torch.testing.assert_close(weights, expanded_weights)


def test_dropout_setting_reaches_grns_and_eval_is_deterministic():
    model = VariableSelectionNetwork(
        num_features=3, hidden_size=8, context_size=5, dropout=0.25
    )
    dropout_layers = [layer for layer in model.modules() if isinstance(layer, nn.Dropout)]
    assert dropout_layers
    assert all(layer.p == 0.25 for layer in dropout_layers)
    assert all(layer.training for layer in dropout_layers)

    model.eval()
    x = torch.randn(2, 6, 3, 8)
    context = torch.randn(2, 5)
    with torch.no_grad():
        first = model(x, context)
        second = model(x, context)
    for before, after in zip(first, second):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_wrong_feature_count_is_rejected():
    model = VariableSelectionNetwork(num_features=4, hidden_size=8)

    with pytest.raises(ValueError, match="Expected 4 features"):
        model(torch.randn(2, 6, 3, 8))
