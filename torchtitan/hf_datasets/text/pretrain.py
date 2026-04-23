# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
from datasets import Dataset, load_dataset

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.base import (
    HFDatasetBase,
    HFDataLoader,
    InterleavedHFDataLoader,
)
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceTextDataset(HFDatasetBase):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        super().__init__(
            dataset=ds,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            dataset_id=dataset_name,
        )

        self._text_processor = text_processor

        # Subclass-owned buffers
        self._inputs_buffer: list[int] = []
        self._positions_buffer: list[int] = []

    def _normalize_positions(self, positions: list[int]) -> list[int]:
        offset = positions[0]
        if offset > 0:
            for i, p in enumerate(positions):
                if p == 0:
                    break
                positions[i] = p - offset
        return positions

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )

                self._inputs_buffer.extend(sample_tokens)
                # Per-document positions reset at document boundaries,
                # matching inference frameworks (e.g. vLLM) that start
                # positions at 0 per request.  Positions wrap at seq_len
                # to stay within the RoPE cache, effectively chunking
                # long documents into seq_len-sized segments.
                # TODO: make overflow policy configurable (chunk / truncate / drop).
                self._positions_buffer.extend(range(len(sample_tokens)))
                self._sample_idx += 1

                while len(self._inputs_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._inputs_buffer[:max_buffer_token_len])
                    pos = torch.LongTensor(
                        self._normalize_positions(
                            self._positions_buffer[:max_buffer_token_len]
                        )
                    )
                    self._inputs_buffer = self._inputs_buffer[max_buffer_token_len:]
                    self._positions_buffer = self._positions_buffer[
                        max_buffer_token_len:
                    ]

                    input = x[:-1]
                    label = x[1:]
                    positions = pos[:-1]
                    yield {"input": input, "positions": positions}, label

            if not self._reloop_or_exhaust():
                break

    def _state_extras(self) -> dict[str, Any]:
        return {
            "inputs_buffer": self._inputs_buffer,
            "positions_buffer": self._positions_buffer,
        }

    def _load_state_extras(self, state_dict: dict[str, Any]) -> None:
        self._inputs_buffer = state_dict["inputs_buffer"]
        if "positions_buffer" not in state_dict:
            logger.warning(
                "Checkpoint missing 'positions_buffer'. Falling back to empty buffer. "
                "RoPE positions may be incorrect with block_causal attention."
            )
        self._positions_buffer = state_dict.get("positions_buffer", [])


class HuggingFaceTextDataLoader(HFDataLoader):
    """Configurable text dataloader that wraps HuggingFaceTextDataset.

    This dataloader can be used for both training and validation by
    configuring the appropriate dataset, seq_len, batch_size, etc.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HFDataLoader.Config):
        dataset: str = "c4_test"
        """Dataset to use"""

    def _build_dataset(
        self,
        source,
        *,
        tokenizer: BaseTokenizer,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        local_batch_size: int,
    ):
        return HuggingFaceTextDataset(
            dataset_name=source.dataset,
            dataset_path=source.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=source.infinite,
        )


@dataclass(kw_only=True, slots=True)
class HFDataSource(HuggingFaceTextDataLoader.Config):
    """Represent one dataset source and its sampling weight"""

    weight: float = 1
    """Data Source sampling weight"""


class InterleavedHuggingFaceTextDataLoader(InterleavedHFDataLoader):
    """Configurable text dataloader that wraps multiple HuggingFaceTextDataset."""

    @dataclass(kw_only=True, slots=True)
    class Config(InterleavedHFDataLoader.Config):
        sources: list[HFDataSource] = field(default_factory=lambda: [HFDataSource()])
        """List of datasources to interleave"""

    def _build_dataset(
        self,
        source,
        *,
        tokenizer: BaseTokenizer,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        local_batch_size: int,
    ):
        return HuggingFaceTextDataset(
            dataset_name=source.dataset,
            dataset_path=source.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=source.infinite,
        )
