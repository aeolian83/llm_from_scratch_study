from __future__ import annotations

from dataclasses import dataclass # 데이터만 담는 클래스를 쉽게 만드는 문법

import torch
from torch import nn

@dataclass
class ToyMLPConfig:
    vocab_size: int
    hidden_size: int
    num_classes: int
    dropout: float

class ToySequenceClassifier(nn.Module):
    def __init__(self, cfg: ToyMLPConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hedden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, cfg.num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")

        x = self.embedding(input_ids)   # (B, T, H) B: batch, T: sequence, H: hidden_size
        x = x.mean(dim=1)               # (B, H)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
