import torch

from llm_lab.models.toy_mlp import ToyMLPConfig, ToySequenceClassifier

def test_toy_model_ouput_shape() -> None:
    cfg = ToyMLPConfig(
        vocab_size=128,
        hidden_size=32,
        num_classes=2,
        dropout=0.1,
    )

    model = ToySequenceClassifier(cfg)
    input_ids = torch.randint(0, 128, (4, 16), dtype=torch.long) # 제일작은 수, 제일 큰수, 데이터 쉐입

    logits = model(input_ids)

    assert logits.shape == (4, 2)
    assert logits.dtype == torch.float32