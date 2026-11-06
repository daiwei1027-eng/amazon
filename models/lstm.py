import torch
import torch.nn as nn
class LSTM(torch.nn.Module):
    def __init__(self, horizon: int,hidden_size: int,num_layers: int,input_size:int=1):
        super().__init__()
        if horizon <= 0 :
            raise ValueError("horizon  must be positive")
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon)
    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        x = past_target
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        output, _ = self.lstm(x)
        last_hidden_state=output[...,-1,:]
        prediction=self.head(last_hidden_state)
        return prediction
