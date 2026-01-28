from typing import List
import math

import torch

from src.layers import Dropout, ReLU

class PositionWiseFeedForward:
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout_p: float = 0.1,
            bias: bool = True,
    ) -> None:
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive.")
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError("dropout_p must be in [0,1)")
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.training = True

        self.w1 = torch.randn(d_model, d_ff) /  math.sqrt(d_model)
        self.w1.requires_grad = True
        self.b1 = None
        if bias:
            self.b1 = torch.zeros(d_ff)
            self.b1.requires_grad = True

        self.w2 = torch.randn(d_ff, d_model) /  math.sqrt(d_ff)
        self.w2.requires_grad = True
        self.b2 = None
        if bias:
            self.b2 = torch.zeros(d_model)
            self.b2.requires_grad = True

        self.activation = ReLU()
        self.dropout = Dropout(p=dropout_p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model, "Last dim must be d_model."

        # [B, L, d_model] -> [B, L, d_ff]
        x = torch.matmul(x, self.w1)
        if self.b1 is not None:
            x = x + self.b1
        
        x = self.activation(x)
        x = self.dropout(x)

        # [B, L, d_ff] -> [B, L, d_model]
        x = torch.matmul(x, self.w2)
        if self.b2 is not None:
            x = x + self.b2
        
        return x

    def parameters(self) -> List[torch.Tensor]:
        params = [self.w1, self.w2]
        if self.b1 is not None:
            params.append(self.b1)
        if self.b2 is not None:
            params.append(self.b2)
        return params
    
    def zero_grad(self) -> None:
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def train(self, mode: bool = True):
        self.training = mode
        self.dropout.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def to(self, device: torch.device):
        self.w1 = self.w1.to(device).detach().requires_grad_(True)
        self.w2 = self.w2.to(device).detach().requires_grad_(True)
        if self.b1 is not None:
            self.b1 = self.b1.to(device).detach().requires_grad_(True)
        if self.b2 is not None:
            self.b2 = self.b2.to(device).detach().requires_grad_(True)
