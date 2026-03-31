from __future__ import annotations

import torch
from lightning import LightningModule
from torch import nn

from llm_lab.models.toy_mlp import ToySequenceClassifier

class ClassificationTask(LightningModule):
    def __init__(
        self,
        model: ToySequenceClassifier,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """
        Batch 데이터 구조
        {
           "input_ids": (B, T),
            "labels": (B,)
        }

        콜러블 함수
        logits = self(input_ids)가 가능한 것은 
        ClassificationTask가 LightningModule을 상속하고, LightningModule이 nn.Module을 상속하는데 
        nn.Module안에 __call__이 forward를 불러 오도록 되어있고, 그렇다면 ClassificationTask에서 제일 먼저 forward메소를 찾는데
        ClassificationTask에 forward가 있기 때문에 불러와 짐

        preds = logits.argmax(dim=-1)
        dim=0 → 세로 방향 (batch 방향)
        dim=1 → 가로 방향 (class 방향)
        dim=-1은 마지막 차원
        (B, C)에서 dim=-1 == dim=1
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits = self(input_ids)                # logits: (B, T) → (B, C)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage=="train"), on_epoch=True)
        self.log(f"{stage}/loss", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdaW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
