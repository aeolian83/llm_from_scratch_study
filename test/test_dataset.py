import torch

from llm_lab.data.dummy_dataset import DummySequenceDataset

def test_dummy_dataset_shapes() -> None:
    dataset = DummySequenceDataset(
        num_samplse=100,
        seq_len=16,
        vocab_size=128,
        threshold=900,
        seed=42,
    )

    sample = dataset[0]

    assert sample["input_ids"].shape == (16,)
    assert sample["labels"].ndim == 0
    assert sample["input_ids"].dtype == torch.long
    assert sample["labels"].dtype == torch.long
    
