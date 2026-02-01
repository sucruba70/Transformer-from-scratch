import math

import torch


class LayerNorm:
    def __init__(
            self,
            normalized_shape: int,
            eps: float = 1e-5,
            elementwise_affine: bool = True,
            bias: bool = True,
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.training = True

        if elementwise_affine:
            self.gamma = torch.ones(self.normalized_shape, requires_grad=True)
            self.beta = None
            if bias:
                self.beta = torch.zeros(self.normalized_shape, requires_grad=True)
        else:
            self.gamma = None
            self.beta = None    

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.normalized_shape
        # 실제 입력 [B, L, dim]
        # [B, L, 1]
        mean = torch.mean(x, dim=-1, keepdim=True) 
        # [B, L, 1]
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        #var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        # [B, L, 1]
        std = torch.sqrt(var + self.eps)

        # [B, L, dim]
        x_norm = (x - mean) / std

        if self.gamma is None:
            y = x_norm
        elif self.beta is None:
            y = self.gamma * x_norm
        else:
            y = self.gamma * x_norm + self.beta
        
        return y
    
    def parameters(self):
        if self.gamma is None:
            return []
        elif self.beta is None:
            return [self.gamma]
        return [self.gamma, self.beta]

    def to(self, device: torch.device):
        if self.gamma is not None:
            self.gamma = self.gamma.to(device).detach().requires_grad_(True)
        if self.beta is not None:
            self.beta = self.beta.to(device).detach().requires_grad_(True)
        return self

    def zero_grad(self) -> None:
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def train(self, mode: bool = True):
        self.training = mode
        return self
    
    def eval(self):
        return self.train(False)
    

class Dropout:
    def __init__(
            self,
            p: float
    ):  
        if not (0.0 <= p < 1.0):
            raise ValueError("p must be in [0,1)")
        self.p = p
        self.training = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        
        keep_prob = 1.0 - self.p
        mask = torch.rand_like(x) < keep_prob
        
        return (x * mask) / keep_prob

    def parameters(self):
        return []

    def zero_grad(self):
        return

    def train(self, mode: bool=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)


class ReLU:
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            x.clamp_(min=0)
            return x
        return x.clamp(min=0)

    def parameters(self):
        return []
    
    def zero_grad(self):
        return

    def train(self, mode: bool=True):
        return self

    def eval(self):
        return self


class Linear:
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        self.weight = torch.randn(in_features, out_features) / math.sqrt(in_features)
        self.weight.requires_grad_()
        self.bias = torch.zeros(out_features, requires_grad=True)

        self.training = True
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, self.weight)
        x = x + self.bias
        return x

    def parameters(self):
        params = [self.weight, self.bias]
        return params
    
    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)
    
    def to(self, device: torch.Tensor):
        self.weight = self.weight.to(device).detach().requires_grad_(True)
        self.bias = self.bias.to(device).detach().requires_grad_(True)
        return self
    
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()