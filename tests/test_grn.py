import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.grn import GRN


@pytest.mark.parametrize(
    "input_shape, output_size, context_shape",
    [
        ((3, 8), 8, None),
        ((2, 5, 8), 8, None),
        ((2, 5, 24), 3, (2, 4)),
        ((2, 5, 8), 4, (2, 5, 6)),
        ((2, 5, 3, 8), 8, (2, 6)),
    ],
    ids=[
        "static", "temporal", "projected_residual", "temporal_context",
        "context_over_time_and_features",
    ],
)
def test_shapes_finite_gradients_and_unchanged_inputs(input_shape, output_size, context_shape):
    torch.manual_seed(42)
    model = GRN(
        input_size=input_shape[-1],
        hidden_size=12,
        output_size=output_size,
        context_size=context_shape[-1] if context_shape is not None else None,
    )
    x = torch.randn(input_shape, requires_grad=True)
    original_x = x.detach().clone()
    context = (
        torch.randn(context_shape, requires_grad=True)
        if context_shape is not None
        else None
    )
    original_context = context.detach().clone() if context is not None else None

    output = model(x, context)

    assert output.shape == input_shape[:-1] + (output_size,)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(x.detach(), original_x, rtol=0, atol=0)
    if context is not None:
        torch.testing.assert_close(context.detach(), original_context, rtol=0, atol=0)

    target = torch.randn_like(output)
    (output - target).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    if context is not None:
        assert context.grad is not None
        assert torch.isfinite(context.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("output_size", [4, 3], ids=["identity_skip", "projected_skip"])
def test_closed_gate_preserves_normalized_residual(output_size):
    model = GRN(input_size=4, hidden_size=8, output_size=output_size)
    x = torch.tensor([[1., 2., 3., 4.], [-2., 3., 1., 0.]])
    with torch.no_grad():
        model.gate[0].weight.zero_()
        model.gate[0].bias.zero_()
        if output_size == 3:
            model.skip.weight.copy_(torch.tensor([
                [1., 0., 0., 1.],
                [0., 2., 0., 0.],
                [0., 0., 3., 0.],
            ]))
            model.skip.bias.copy_(torch.tensor([0.5, -1., 2.]))

    residual = (
        x if output_size == 4
        else torch.tensor([[5.5, 3., 11.], [-1.5, 5., 5.]])
    )
    expected = F.layer_norm(residual, (output_size,))

    torch.testing.assert_close(model(x), expected)


def test_static_context_broadcast_matches_individual_timesteps():
    torch.manual_seed(7)
    model = GRN(input_size=4, hidden_size=8, output_size=6, context_size=3)
    model.eval()
    x = torch.randn(2, 5, 4)
    context = torch.randn(2, 3)

    with torch.no_grad():
        together = model(x, context)
        individual = torch.stack([
            model(x[:, step, :], context)
            for step in range(x.shape[1])
        ], dim=1)

    torch.testing.assert_close(together, individual)


def test_omitted_context_matches_zero_context():
    model = GRN(input_size=4, hidden_size=8, output_size=4, context_size=3)
    model.eval()
    x = torch.randn(2, 5, 4)

    with torch.no_grad():
        without_context = model(x)
        with_zero_context = model(x, torch.zeros(2, 3))

    torch.testing.assert_close(without_context, with_zero_context)


def test_context_requires_configured_projection():
    model = GRN(input_size=4, hidden_size=8, output_size=4)

    with pytest.raises(ValueError, match="context_size must be configured"):
        model(torch.randn(2, 5, 4), torch.randn(2, 3))


def test_dropout_setting_and_training_eval_behavior():
    torch.manual_seed(11)
    model = GRN(input_size=8, hidden_size=12, output_size=8, dropout=0.25)
    dropout_layers = [layer for layer in model.modules() if isinstance(layer, nn.Dropout)]
    assert dropout_layers
    assert all(layer.p == 0.25 for layer in dropout_layers)
    x = torch.randn(16, 8)

    model.train()
    with torch.no_grad():
        torch.manual_seed(20)
        first_train = model(x)
        torch.manual_seed(21)
        second_train = model(x)
    assert not torch.allclose(first_train, second_train)

    model.eval()
    with torch.no_grad():
        torch.manual_seed(20)
        first_eval = model(x)
        torch.manual_seed(21)
        second_eval = model(x)
    torch.testing.assert_close(first_eval, second_eval, rtol=0, atol=0)
