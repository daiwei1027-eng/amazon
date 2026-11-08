import torch.nn as nn
import torch
from models.grn import GRN
from models.vsn import VariableSelectionNetwork
from models.attention import InterpretableMultiHeadAttention
class TFT(nn.Module):
    def __init__(
        self,
        *,
        horizon: int,
        num_observed_features: int,
        num_known_features: int,
        static_cardinalities: dict[str, int],
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_heads:int=1
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.static_features = list(static_cardinalities)
        self.static_embeddings = nn.ModuleDict({
            name: nn.Embedding(count, hidden_size)
            for name, count in static_cardinalities.items()
        })
        self.observed_projections = nn.ModuleList([
            nn.Linear(1, hidden_size)
            for _ in range(num_observed_features)
        ])

        self.known_projections = nn.ModuleList([
            nn.Linear(1, hidden_size)
            for _ in range(num_known_features)
        ])
        self.static_vsn = VariableSelectionNetwork(
            len(self.static_features),
            hidden_size,
            dropout=dropout,
        )

        self.selection_context = GRN(
            hidden_size, hidden_size, hidden_size,
            dropout=dropout,
        )

        self.past_vsn = VariableSelectionNetwork(
            num_observed_features + num_known_features,
            hidden_size,
            context_size=hidden_size,
            dropout=dropout,
        )

        self.future_vsn = VariableSelectionNetwork(
            num_known_features,
            hidden_size,
            context_size=hidden_size,
            dropout=dropout,
        )
        self.initial_hidden_grn = GRN(
            hidden_size, hidden_size, hidden_size,
            dropout=dropout,
        )

        self.initial_cell_grn = GRN(
            hidden_size, hidden_size, hidden_size,
            dropout=dropout,
        )

        lstm_config = {
            "input_size": hidden_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "dropout": dropout if num_layers > 1 else 0.0,
        }
        self.encoder_lstm = nn.LSTM(**lstm_config)
        self.decoder_lstm = nn.LSTM(**lstm_config)

        self.post_lstm_gate = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GLU(dim=-1),
        )
        self.post_lstm_norm = nn.LayerNorm(hidden_size)

        self.enrichment_context = GRN(
            hidden_size, hidden_size, hidden_size,
            dropout=dropout,
        )

        self.static_enrichment = GRN(
            hidden_size, hidden_size, hidden_size,
            context_size=hidden_size,
            dropout=dropout,
        )
        self.attention = InterpretableMultiHeadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.post_attention_gate = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GLU(dim=-1),
        )
        self.post_attention_norm = nn.LayerNorm(hidden_size)

        self.positionwise_grn = GRN(
            hidden_size, hidden_size, hidden_size,
            dropout=dropout,
        )

        self.output_gate = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GLU(dim=-1),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

        self.head = nn.Linear(hidden_size, 1)

    def project_features(self, x, projections):
        projected = [
            layer(x[..., i:i + 1])
            for i, layer in enumerate(projections)
        ]
        return torch.stack(projected, dim=-2)

    def forward(self, x):
        if x["future_known"].shape[1] != self.horizon:
            raise ValueError("Future input length must equal horizon")
        observed = self.project_features(
            x["past_observed"], self.observed_projections
        )
        known_past = self.project_features(
            x["past_known"], self.known_projections
        )
        known_future = self.project_features(
            x["future_known"], self.known_projections
        )
        static_embedded = torch.stack(
            [
                self.static_embeddings[name](x["static"][:, i])
                for i, name in enumerate(self.static_features)
            ],
            dim=1,
        )
        past_embedded = torch.cat(
            [observed, known_past],
            dim=-2,
        )
        static_selected, static_weights = self.static_vsn(
            static_embedded
        )
        context = self.selection_context(static_selected)

        past_selected, past_weights = self.past_vsn(
            past_embedded, context
        )
        future_selected, future_weights = self.future_vsn(
            known_future, context
        )
        h0 = self.initial_hidden_grn(static_selected)
        c0 = self.initial_cell_grn(static_selected)

        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)

        past_output, (h, c) = self.encoder_lstm(
            past_selected, (h0, c0)
        )
        future_output, _ = self.decoder_lstm(
            future_selected, (h, c)
        )
        lstm_output = torch.cat(
            [past_output, future_output], dim=1
        )

        lstm_input = torch.cat(
            [past_selected, future_selected], dim=1
        )

        temporal = self.post_lstm_norm(
            lstm_input + self.post_lstm_gate(lstm_output)
        )

        enrichment_context = self.enrichment_context(
            static_selected
        )
        enriched = self.static_enrichment(
            temporal, enrichment_context
        )
        attention_output, attention_weights = self.attention(enriched)
        attention_processed = self.post_attention_norm(
            enriched + self.post_attention_gate(attention_output)
        )

        feedforward = self.positionwise_grn(attention_processed)

        final_features = self.output_norm(
            temporal + self.output_gate(feedforward)
        )

        future_hidden = final_features[:, -self.horizon:, :]
        prediction = self.head(future_hidden).squeeze(-1)

        return prediction

