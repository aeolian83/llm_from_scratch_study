from omegaconf import OmegaConf

from llm_lab.train import build_datamodule, build_model

def test_builds_from_config() -> None:
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "model": {
                "vocab_size": 128,
                "hidden_size": 64,
                "num_classes": 2,
                "dropout": 0.1,
            },
            "data": {
                "num_samples": 128,
                "seq_len": 16,
                "vocab_size": 128,
                "threshold": 900,
                "train_ratio": 0.8,
                "batch_size": 16,
                "num_workers":0, 
                "pin_memory": False,
            },
        }
    )

    model = build_model(cfg)
    datamodule = build_datamodule(cfg)

    assert model is not None
    assert datamodule is not None