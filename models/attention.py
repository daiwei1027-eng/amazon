import torch
import torch.nn as nn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=1, dropout=0.0):
        super().__init__()

        if num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by a positive num_heads"
            )

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # 所有头共享同一个 V
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.head_dim, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, steps, _ = x.shape

        q = self.q_proj(x).reshape(
            batch, steps, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).reshape(
            batch, steps, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).unsqueeze(1)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = torch.ones(
            steps, steps,
            dtype=torch.bool,
            device=x.device,
        ).triu(1)

        weights = scores.masked_fill(
            mask, float("-inf")
        ).softmax(dim=-1)

        attended = (self.dropout(weights) @ v).mean(dim=1)

        return self.out_proj(attended), weights.mean(dim=1)