from typing import Optional
import math

import torch


class MultiHeadAttention:
    def __init__(
        self,
        d_model: int,
        heads: int,
        bias: bool = True
    ):
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads.")
        
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.bias = bias
        self.training = True

        self.scale = math.sqrt(d_model)
        self.w_q = torch.randn(d_model, d_model) / self.scale
        self.w_q.requires_grad_()
        self.w_k = torch.randn(d_model, d_model) / self.scale
        self.w_k.requires_grad_()
        self.w_v = torch.randn(d_model, d_model) / self.scale
        self.w_v.requires_grad_()
        self.w_o = torch.randn(d_model, d_model) / self.scale
        self.w_o.requires_grad_()

        if bias:
            self.b_q = torch.zeros(d_model, requires_grad=True)
            self.b_k = torch.zeros(d_model, requires_grad=True)
            self.b_v = torch.zeros(d_model, requires_grad=True)
            self.b_o = torch.zeros(d_model, requires_grad=True)
        else:
            self.b_q = self.b_k = self.b_v = self.b_o = None  
    
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(query, key, value, key_padding_mask, attn_mask)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # Q, K, V 생성 (각각은 [B, max_len, d_model])
        # [B, max_len, d_model] * [d_model, d_model] -> [B, max_len, d_model]
        q = torch.matmul(query, self.w_q)
        if self.b_q is not None:
            q = q + self.b_q
        k = torch.matmul(key, self.w_k)
        if self.b_k is not None:
            k = k + self.b_k
        v = torch.matmul(value, self.w_v)
        if self.b_v is not None:
            v = v + self.b_v
        
        # q,k,v를 multihead로 reshape
        # [B, max_len, d_model] -> [B, max_len, heads, d_heads] -> [B, heads, max_len, d_heads]
        q = q.view(batch_size, -1, self.heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, -1, self.heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, -1, self.heads, self.head_dim).transpose(1,2)

        # attention_score
        # [B, heads, max_len, d_heads] @ [B, heads, d_heads, max_len] -> [B, heads, max_len, max_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Mask
        # [L_q, L_k] -> [1, 1, L_q, L_k]
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None, :, :]
        
        # padding
        # [B,L] -> [B, 1, 1, L] 
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 2:
                key_padding_mask = key_padding_mask[:, None, None, :]

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask * key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # attention weights
        # [B, heads, max_len, max_len] @ [B, heads, max_len, d_heads] -> [B, heads, max_len, d_heads]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        # Concationate
        # [B, heads, max_len, d_heads] -> [B, max_len, heads, d_heads] -> [B, max_len, d_model]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        output = torch.matmul(context, self.w_o)
        if self.b_o is not None:
            output = output + self.b_o

        return output

    def parameters(self):
        params = [self.w_q, self.w_k, self.w_v, self.w_o]
        if self.b_q is not None:
            params.extend([self.b_q, self.b_k, self.b_v, self.b_o])
        return params
    
    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)
    
    def to(self, device: torch.device):
        self.w_q = self.w_q.to(device).detach().requires_grad_(True)
        self.w_k = self.w_k.to(device).detach().requires_grad_(True)
        self.w_v = self.w_v.to(device).detach().requires_grad_(True)
        self.w_o = self.w_o.to(device).detach().requires_grad_(True)

        if self.b_q is not None:
            self.b_q = self.b_q.to(device).detach().requires_grad_(True)
            self.b_k = self.b_k.to(device).detach().requires_grad_(True)
            self.b_v = self.b_v.to(device).detach().requires_grad_(True)
            self.b_o = self.b_o.to(device).detach().requires_grad_(True)
        
        return self
    
    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()