from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

@dataclass
class MinimalGPTConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    norm_type: str = "layernorm"

class MinimalGPT(nn.Module):
    def __init__(self, cfg: MinimalGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")
        
        batch_size, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceed max_seq_len={self.cfg.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)    # 1차원을 2차원으로 변환 대신 원래 배열을 가장 마지막 차원으로 한다.
        positions = positions.expand(batch_size, seq_len)    # 다만 이전에서 만든 데이터 차원이 이 코드줄에서 만든 seq_len이랑 다를경우 어떻게 된느지 알고 싶다. 

        tok = self.token_embedding(input_ids)    # (B, T, C) batch, seq_len, hidden_size
        pos = self.position_embedding(positions) # (B, T, C)

        x = self.dropout(tok + pos)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits