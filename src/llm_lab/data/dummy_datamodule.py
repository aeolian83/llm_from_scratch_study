from __future__ import annotations

from dataclasses import dataclass

from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from llm_lab.data.dummy_dataset import DummySequenceDataset

@dataclass
class DummyDataConfig:
    num_samples: int
    seq_len: int
    vocab_size: int
    threshold: int
    train_ratio: float
    batch_size: int
    num_workers: int
    pin_memory: bool


class DummySequenceDataModule(LightningDataModule):
    def __init__(self, cfg: DummyDataConfig, seed: int = 42) -> None:
        super().__init__()
        self.cfg = cfg
        self.seed = seed
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        full_dataset =  DummySequenceDataset(
            num_samples=self.cfg.num_samples,
            seq_len=self.cfg.seq_len,
            vocab_size=self.cfg.vocab_size,
            threshold=self.cfg.threshold,
            seed=self.cfg.seed,
        )

        train_size = int(len(full_dataset) * self.cfg.train_ratio)
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
        )

    def train_dataloader(self) -> DataLoader:
        """
        학습용 DataLoader를 반환한다.

        Note:
            - pin_memory=True이면 CPU → GPU 데이터 전송 속도가 향상된다.
            - GPU 학습 시 성능 최적화를 위해 사용하는 옵션이다.
        """
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before train_dataloader().")

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            # 페이저블 메모리와 반대의미로 핀메모리이다. 페이저블은 cpu에서 gpu로 가며 복사라는 한단계를 거치지만, 핀메모리를 설정하면, GPU가 직접
            # 메모리에 접근 할 수 있어서 속도를 향상시켜준다. 
            pin_memory=self.cfg.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataloader is None:
            raise RuntimeError("DataModule.setup() must be called before val_dataloader().")

        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )