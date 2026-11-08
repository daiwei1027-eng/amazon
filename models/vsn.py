import torch
import torch.nn as nn

from models.grn import GRN


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        *,
        context_size: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.weight_grn = GRN(
            input_size=num_features * hidden_size,
            hidden_size=hidden_size,
            output_size=num_features,
            context_size=context_size,
            dropout=dropout,
        )

        self.feature_grns = nn.ModuleList([
            GRN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
            for _ in range(num_features)
        ])

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_features = len(self.feature_grns)
        if x.ndim < 2 or x.shape[-2] != num_features:
            raise ValueError(
                f"Expected {num_features} features on the second-to-last dimension"
            )

        flattened = x.flatten(start_dim=-2)
        weights = self.softmax(self.weight_grn(flattened, context))

        transformed_features = torch.stack(
            [
                grn(x[..., i, :])
                for i, grn in enumerate(self.feature_grns)
            ],
            dim=-2,
        )
        selected_features = (
            transformed_features * weights.unsqueeze(-1)
        ).sum(dim=-2)

        return selected_features, weights
