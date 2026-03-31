from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

@dataclass
class DummySequenceSample:
    input_ids: torch.Tensor
    label: torch.Tensor

class DummySequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        threshold: int,
        seed: int = 42,
    ) -> None:
        """
        랜덤한 토큰 시퀀스를 생성하고, 각 시퀀스의 토큰 합을 기준으로
        이진 분류(label)를 생성하는 더미 데이터셋을 초기화한다.

        이 Dataset은 실제 텍스트 데이터 없이 모델 구조 및 학습 파이프라인을
        테스트하기 위한 synthetic 데이터셋이다.

        Args:
            num_samples (int): 생성할 데이터 샘플의 개수(전체 데이터셋 크기)
            seq_len (int): 각 샘플의 시퀀스 길이 (토큰 개수).
            vocab_size (int): 토큰의 종류 개수. 각 토큰은 [0, vocab_size) 범위의 정수로 생성된다.
            threshold (int): 각 시퀀스의 토큰 합(token sum)이 이 값 이상이면 label=1, 그렇지 않으면 label=0으로 설정된다.
            seed (int, optional): 랜덤 시드 값. 동일한 seed를 사용하면 항상 동일한 데이터가 생성된다. 기본값은 42.

        Attributes:
            input_ids (torch.Tensor):
                랜덤하게 생성된 토큰 시퀀스.
                - dtype: torch.long (int64)
                - shape: (num_samples, seq_len)
                - 값 범위: [0, vocab_size)
                예:
                    tensor([
                        [1, 5, 3, 2],
                        [0, 2, 4, 1],
                        ...
                    ])

            labels (torch.Tensor):
                각 시퀀스의 토큰 합이 threshold 이상인지 여부를 나타내는 이진 레이블.
                - dtype: torch.long (int64)
                - shape: (num_samples,)
                - 값:
                    1 → token_sum >= threshold
                    0 → token_sum < threshold
                예:
                    tensor([1, 0, 1, ...])

        Raises:
            ValueError:
                - vocab_size <= 1 인 경우
                - seq_len <= 0 인 경우
                - num_samples <= 0 인 경우
        """

        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1.")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")            

        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_samples, seq_len),
            dtype=torch.long,
            generator=generator,
        )
        token_sums = self.input_ids.sum(dim=1)
        self.label = (torken_sums >= threshold).to(torch.long)

    def __len__(self) -> int:
        """
        데이터셋에 포함된 전체 샘플의 개수를 반환한다.

        이 값은 DataLoader에서 반복(iteration) 횟수를 결정하는 기준으로 사용되며,
        각 인덱스(idx)는 하나의 시퀀스 샘플을 의미한다.

        Returns:
            int:
                데이터셋의 총 샘플 개수 (num_samples)

        Note:
            - self.input_ids의 shape은 (num_samples, seq_len)이다.
            - 따라서 size(0)은 샘플 개수(num_samples)를 의미한다.
            - 시퀀스 길이(seq_len)는 size(1)로 확인할 수 있다.
        """
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        주어진 인덱스의 단일 샘플을 반환한다.

        Args:
            index (int): 샘플 인덱스

        Returns:
            dict[str, torch.Tensor]:
                - input_ids: (seq_len,) torch.long
                - labels: () torch.long (이진 레이블)
        """
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
        }