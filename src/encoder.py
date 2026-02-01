from typing import Optional

import torch

from src.attention import MultiHeadAttention
from src.feedforward import PositionWiseFeedForward
from src.layers import LayerNorm, Dropout
from src.embeddings import Embedding, PositionalEncoding


class EncoderLayer:
    def __init__(
            self,
            d_model: int,
            heads: int,
            d_ff: int,
            dropout_p: float = 0.1,
            ln_bias: bool = True,
            attn_bias: bool = True,
            ffn_bias: bool = True,
            elementwise_affine: bool = True,
            eps: float = 1e-5,
    ):
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.ln_bias = ln_bias
        self.attn_bias = attn_bias
        self.ffn_bias = ffn_bias
        self.eps = eps
        self.training = True

        self.self_attn = MultiHeadAttention(d_model=d_model, heads=heads, bias=attn_bias)
        
        self.norm1 = LayerNorm(
            normalized_shape=d_model,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=ln_bias,
        )
        
        self.dropout1 = Dropout(p=dropout_p)
        
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_p=dropout_p, bias=ffn_bias)
        
        self.norm2 = LayerNorm(
            normalized_shape=d_model,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=ln_bias,
        )
        
        self.dropout2 = Dropout(p=dropout_p)

    def __call__(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(x, mask)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None

    ) -> torch.Tensor:
        # Multi-Head Attention
        # [B, L, d_model] -> [B, L, d_model]
        residual = x
        attn_output = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask,
            attn_mask=None
        )

        # Add & Norm
        x = self.norm1(residual + self.dropout1(attn_output))

        # FFN
        # [B, L, d_model] -> [B, L, d_ff] -> [B, L, d_model]
        residual = x
        ffn_output = self.ffn(x)
        
        # Add & Norm
        x = self.norm2(residual + self.dropout2(ffn_output))

        return x
    
    def parameters(self):
        params = []
        params.extend(self.self_attn.parameters())
        params.extend(self.norm1.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.norm2.parameters())

        return params

    def zero_grad(self):
        self.self_attn.zero_grad()
        self.norm1.zero_grad()
        self.ffn.zero_grad()
        self.norm2.zero_grad()
    
    def train(self, mode: bool = True):
        self.training = mode
        self.self_attn.train(mode)
        self.dropout1.train(mode)
        self.ffn.train(mode)
        self.dropout2.train(mode)
        self.norm1.train(mode)
        self.norm2.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def to(self, device: torch.device):
        self.self_attn.to(device)
        self.norm1.to(device)
        self.ffn.to(device)
        self.norm2.to(device)
        return self


class Encoder:
    def __init__(
        self,
        num_embeddings: int = 30000,
        d_model: int = 512,
        max_len: int = 512,
        heads: int = 8,
        d_ff: int = 2048,
        dropout_p: float = 0.1,
        ln_bias: bool = True,
        attn_bias: bool = True,
        ffn_bias: bool = True,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        num_layers: int = 6,
    ):
        self.embedding = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=d_model,
            padding_idx = 0,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout_p
        )
        self.layers = [
            EncoderLayer(
                d_model=d_model,
                heads=heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
                ln_bias=ln_bias,
                attn_bias=attn_bias,
                ffn_bias=ffn_bias,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )
            for _ in range(num_layers)
        ]
        self.training = True
    
    def __call__(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(x, mask)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. 임베딩에 넣기
        # 2. Positional encoding 하기
        # 3. Embedding + PE 하기
        # 4. self.layers에 넣어서 쭉쭉 들어가기
        # 5. 마지막에 output 내보내기.

        # [B, L] -> [B, 1, 1, L]
        if mask is not None and mask.dim() == 2:
            mask = mask[:, None, None, :]

        # x: [B, L] -> [B, L, d_model]
        x = self.embedding(x)
        # [B, L, d_model] -> [B, L, d_model]
        x = self.positional_encoding(x)
        # [B, L, d_model] -> [B, L, d_model]
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def train(self, mode: bool = True):
        self.training = mode
        self.embedding.train(mode)
        self.positional_encoding.train(mode)
        for layer in self.layers:
            layer.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def to(self, device: torch.device):
        self.embedding.to(device)
        self.positional_encoding.to(device)
        for layer in self.layers:
            layer.to(device)
        return self