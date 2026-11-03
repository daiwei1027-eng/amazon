import torch
class NaiveBaseline(torch.nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon

    def forward(
            self,
            past_target: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = past_target.shape[0]

        output = past_target.new_zeros(
            (batch_size, self.horizon)
        )
        last_value = past_target[:, -1]

        for i in range(self.horizon):
            output[:, i] = last_value

        return output
