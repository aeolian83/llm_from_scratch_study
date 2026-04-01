from __future__ import annotations

from typing import Any
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.logger import Logger

try:
    from lightning.pytorch.loggers import WandbLogger
except Exception:
    WandbLogger = None

def build_logger(cfg: Any) -> Logger:
    logger_name = cfg.logger.name

    if logger_name == "tensorboard":
        return TensorBoardLogger(
            save_dir=cfg.logger.save_dir,
            name=cfg.experiment.name 
        )

    if logger_name == "wandb":
        if WandbLogger in None:
            raise ImportError("wandb logger requested, but wandb is not installed.")
        return WandbLogger(
            project=cfg.logger.project,
            save_dir=cfg.logger.save_dir,
            name=cfg.experimet.name,
            offline=cfg.logger.offline,
        )

    raise ValueError(f"Unsupported logger type: {logger_name}")