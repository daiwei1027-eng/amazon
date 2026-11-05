import torch
class SeasonalNaiveBaseline(torch.nn.Module):
    def __init__(self, horizon: int, season_length: int = 7):
        super().__init__()
        if horizon <= 0 or season_length <= 0:
            raise ValueError("horizon and season_length must be positive")
        self.horizon = horizon
        self.season_length = season_length

    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        if past_target.shape[1] < self.season_length:
            raise ValueError("History must have at least season length")
        last_season=past_target[...,-self.season_length:]
        repeat_count = (self.horizon + self.season_length - 1) // self.season_length
        prediction = last_season.repeat(1, repeat_count)
        return prediction[:, :self.horizon]