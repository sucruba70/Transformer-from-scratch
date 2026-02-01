import math
from typing import Optional

import torch

from src.attention import MultiHeadAttention
from src.feedforward import PositionWiseFeedForward
from src.layers import LayerNorm, Dropout
from src.embeddings import Embedding, PositionalEncoding


class DecoderLayer:
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
        self.training = True

        self.masked_attn = MultiHeadAttention(d_model=d_model, heads=heads, bias=attn_bias)
        self.norm1 = LayerNorm(
            normalized_shape=d_model,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=ln_bias,
        )
        self.dropout1 = Dropout(p=dropout_p)

        self.cross_attn = MultiHeadAttention(d_model=d_model, heads=heads, bias=attn_bias)
        self.norm2 = LayerNorm(
            normalized_shape=d_model,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=ln_bias,
        )
        self.dropout2 = Dropout(p=dropout_p)

        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_p=dropout_p, bias=ffn_bias)
        self.norm3 = LayerNorm(
            normalized_shape=d_model,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=ln_bias,
        )
        self.dropout3 = Dropout(p=dropout_p)

    def __call__(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None,
            tgt_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, memory, tgt_key_padding_mask, memory_key_padding_mask, tgt_attn_mask)
    
    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None,
            tgt_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # Masked Multi-Head Attention
        residual = x
        masked_attn_output = self.masked_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_attn_mask,
        )
        # Add & Norm
        x = self.norm1(residual + self.dropout1(masked_attn_output))
        
        # Cross Multi-Head Attention
        residual = x
        cross_attn_output = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=None,
        )
        # Add & Norm
        x = self.norm2(residual + self.dropout2(cross_attn_output))

        # FFN
        residual = x
        ffn_output = self.ffn(x)
        # Add & Norm
        x = self.norm3(residual + self.dropout3(ffn_output))

        return x
    
    def parameters(self):
        params = []
        params.extend(self.masked_attn.parameters())
        params.extend(self.norm1.parameters())
        params.extend(self.cross_attn.parameters())
        params.extend(self.norm2.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.norm3.parameters())
        return params

    def train(self, mode: bool = True):
        self.training = mode

        self.masked_attn.train(mode)
        self.norm1.train(mode)
        self.dropout1.train(mode)

        self.cross_attn.train(mode)
        self.norm2.train(mode)
        self.dropout2.train(mode)

        self.ffn.train(mode)
        self.norm3.train(mode)
        self.dropout3.train(mode)
        return self
    
    def eval(self):
        return self.train(False)

    def to(self, device: torch.device):
        self.masked_attn.to(device)
        self.norm1.to(device)
        self.cross_attn.to(device)
        self.norm2.to(device)
        self.ffn.to(device)
        self.norm3.to(device)
        return self
    
    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()


class Decoder:
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
            padding_idx=0,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout_p
        )

        self.layers = [
            DecoderLayer(
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
    
    def __call__(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, memory, tgt_key_padding_mask, memory_key_padding_mask, tgt_attn_mask)

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None,
            tgt_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Embedding
        x = self.embedding(x)
        # Positional Encoding
        x = self.positional_encoding(x)

        # Decoder Layer
        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_attn_mask=tgt_attn_mask
            )

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