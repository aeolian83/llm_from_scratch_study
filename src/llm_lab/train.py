from __future__ import annotations

import hydra
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf

from llm_lab.data.dummy_datamodule import DummyDataConfig, DummySequenceDataModule
from llm_lab.lightning.task import ClassificationTask
from llm_lab.models.toy_mlp import ToyMLPConfig, ToySequenceClassifier
from llm_lab.utils.logging_utils import build_logger
from llm_lab.utils.seed import seed_everything_local

def build_model(cfg: DictConfig) -> ToySequenceClassifier:
    model_cfg = ToyMLPConfig(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )
    return ToySequenceClassifier(model_cfg)

def build_datamodule(cfg: DictConfig) -> DummySequenceDataModule:
    data_cfg = DummyDataConfig(
        num_samples=cfg.data.num_samples,
        seq_len=cfg.data.seq_len,
        vocab_size=cfg.data.vocab_size,
        threchold=cfg.data.threchold,
        train_ratio=cfg.data.train_ratio,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_worker,
        pin_memory=cfg.data.pin_memory,
    )
    return DummySequenceDataModule(data_cfg, seed=cfg.seed)

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    seed_everything_local(cfg.seed)

    model = build_model(cfg)
    datamodule = build_datamodule(cfg)

    task = ClassificationTask(
        model=model,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.opimizer.weight_decay,
    )

    logger = build_logger(cfg)

    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
    )