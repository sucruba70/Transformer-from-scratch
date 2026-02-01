import math
from typing import Optional

import torch

from src.encoder import Encoder
from src.decoder import Decoder
from src.layers import Linear

class Transformer:
    def __init__(
        self,
        encoder_num_embeddings: int = 30000,
        decoder_num_embeddings: int = 30000,
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
        self.encoder = Encoder(
            num_embeddings=encoder_num_embeddings,
            d_model=d_model,
            max_len=max_len,
            heads=heads,
            d_ff=d_ff,
            dropout_p=dropout_p,
            num_layers=num_layers,   
        )
        self.decoder = Decoder(
            num_embeddings=decoder_num_embeddings,
            d_model=d_model,
            max_len=max_len,
            heads=heads,
            d_ff=d_ff,
            dropout_p=dropout_p,
            num_layers=num_layers,
        )
        self.linear = Linear(d_model, decoder_num_embeddings)
        self.training = True
    
    def __call__(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(
            src_ids=src_ids,
            tgt_ids=tgt_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, L_tgt = tgt_ids.shape

        tgt_attn_mask = torch.tril(
            torch.ones(L_tgt, L_tgt, device=tgt_ids.device, dtype=torch.long)
        )

        memory = self.encoder(src_ids, key_padding_mask=src_key_padding_mask)
        
        x = self.decoder(
            tgt_ids,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_attn_mask=tgt_attn_mask,
        )

        logits = self.linear(x)
        return logits
    
    def parameters(self):
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        params.extend(self.linear.parameters())
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def train(self, mode: bool = True):
        self.training = mode
        self.encoder.train(mode)
        self.decoder.train(mode)
        self.linear.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def to(self, device: torch.device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.linear.to(device)
        return self