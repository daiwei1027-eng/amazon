import torch
import torch.nn as nn
class LSTM(torch.nn.Module):
    def __init__(self, horizon: int,hidden_size: int,num_layers: int):
        super().__init__()
        if horizon <= 0 :
            raise ValueError("horizon  must be positive")
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon)
    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        input=past_target.unsqueeze(-1)
        output,__=self.lstm(input)
        last_hidden_state=output[...,-1,:]
        prediction=self.head(last_hidden_state)
        return prediction
