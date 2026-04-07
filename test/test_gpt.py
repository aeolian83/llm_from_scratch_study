import torch

from llm_lab.models.gpt import MinimalGPT, MinimalGPTConfig

def test_minimal_gpt_output_shape() -> None:
    cfg = MinimalGPTConfig(
        vocab_size=128,
        max_seq_len=32,
        hidden_size=64,
        num_layers=0,
        num_heads=4,
        dropout=0.0,
    )
    model = MinimalGPT(cfg)

    input_ids = torch.randint(0, 128, (4, 16), dtype=torch.long)
    logits = model(input_ids)

    assert logits.shape == (4, 16, 128)
    assert logits.dtype == torch.float32
