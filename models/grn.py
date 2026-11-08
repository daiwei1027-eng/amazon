import torch.nn as nn
import torch


class GRN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        context_size: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context_projection = (
            nn.Linear(context_size, hidden_size, bias=False)
            if context_size is not None
            else None
        )
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Linear(output_size, output_size * 2),

            nn.GLU(dim=-1),
        )

        self.skip = (
            nn.Identity()
            if input_size == output_size
            else nn.Linear(input_size, output_size)
        )
        self.norm = nn.LayerNorm(output_size)

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.fc1(x)

        if context is not None:
            if self.context_projection is None:
                raise ValueError(
                    "context_size must be configured when context is provided"
                )

            projected_context = self.context_projection(context)

            while projected_context.ndim < hidden.ndim:
                projected_context = projected_context.unsqueeze(-2)

            hidden = hidden + projected_context

        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.gate(hidden)

        return self.norm(residual + hidden)
