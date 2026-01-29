import torch

from src.layers import Dropout


class PositionalEncoding:
    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        self.d_model = d_model
        self.max_len = max_len
        self.training = True

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [max_len, d_model] -> [1, max_len, d_model]
        self.pe = pe.unsqueeze(0)
        self.dropout = Dropout(p=dropout)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len, :]

        if self.dropout is not None:
            x = self.dropout(x)
        
        return x

    def train(self, mode: bool = True):
        self.training = mode
        self.dropout.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def parameters(self):
        return []
    
    def zero_grad(self):
        return

    def to(self, device):
        self.pe = self.pe.to(device)
        return self