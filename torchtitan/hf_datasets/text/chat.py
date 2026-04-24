# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

import torch
import tyro
from datasets import Dataset, load_dataset

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.base import (
    HFDatasetBase,
    HFDataLoader,
    InterleavedHFDataLoader,
)
from torchtitan.tools.logging import logger


class ChatDataset(HFDatasetBase):
    """Dataset for single-turn chat/instruction-tuning.

    Tokenizes [user, assistant] message pairs, masks prompt tokens with
    IGNORE_INDEX in labels, and uses greedy sequence packing with
    per-document positions. Implements Stateful for checkpointing.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: BaseTokenizer,
        sample_processor: Callable,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        if tokenizer.eos_id is None:
            raise ValueError(
                "Tokenizer does not have an eos_id set. "
                "ChatDataset requires a tokenizer with a valid EOS token."
            )

        dataset_id = f"{dataset.info.dataset_name}/{dataset.split}"

        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            dataset_id=dataset_id,
        )

        self._eos_id = tokenizer.eos_id
        self._sample_processor = sample_processor

        # Subclass-owned buffers
        self._inputs_buffer: list[int] = []
        self._labels_buffer: list[int] = []
        self._positions_buffer: list[int] = []
        self._pending_input_ids: list[int] = []
        self._pending_label_ids: list[int] = []

        self._logged_first_sample = False

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        """Validate that messages are a single-turn [user, assistant] pair."""
        # TODO: expand this to multi-turn
        if len(messages) != 2:
            raise ValueError(
                f"Expected single-turn [user, assistant], got {len(messages)} messages"
            )
        if messages[0]["role"] != "user":
            raise ValueError(
                f"First message must be 'user', got '{messages[0]['role']}'"
            )
        if messages[1]["role"] != "assistant":
            raise ValueError(
                f"Second message must be 'assistant', got '{messages[1]['role']}'"
            )

    def _tokenize_sample(
        self, sample: dict[str, Any]
    ) -> tuple[list[int], list[int]] | None:
        """Tokenize a single-turn sample and create input/label pairs.

        Returns (input_ids, label_ids) where input_ids = tokens[:-1] and
        label_ids = tokens[1:] with prompt tokens masked as IGNORE_INDEX.
        Returns None if the sample exceeds seq_len (dropped to avoid
        training on truncated responses).

        Uses incremental prefix re-tokenization to find the prompt/response
        token boundary, avoiding BPE merge errors.
        """
        messages = self._sample_processor(sample)
        self._validate_messages(messages)

        full_text = self._tokenizer.apply_chat_template(messages)
        full_text = full_text.rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._eos_id:
            full_tokens.append(self._eos_id)

        if not self._logged_first_sample:
            logger.info(f"[ChatDataset] First sample full:\n{full_text}")
            self._logged_first_sample = True

        if len(full_tokens) - 1 > self.seq_len:
            logger.debug(
                f"Dropping sample {self._sample_idx}: "
                f"tokens exceeds seq_len {self.seq_len}"
            )
            return None

        input_ids = full_tokens[:-1]
        label_ids = full_tokens[1:]

        prompt_text = self._tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        prompt_tokens = self._tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        prompt_len = len(prompt_tokens)

        mask_end = min(max(prompt_len - 1, 0), len(label_ids))
        label_ids[:mask_end] = [IGNORE_INDEX] * mask_end

        return input_ids, label_ids

    def __iter__(self):
        yield from self._iter_greedy_packed()

    def _iter_greedy_packed(self):
        """Greedy packing: pack examples sequentially until seq_len is full.
        Document boundaries are marked by EOS tokens between packed examples.
        The model's flex/varlen attention mask uses these EOS positions to
        prevent cross-document attention.
        """
        if self._pending_input_ids:
            input_ids = self._pending_input_ids
            label_ids = self._pending_label_ids
            self._pending_input_ids = []
            self._pending_label_ids = []
            self._inputs_buffer.extend(input_ids)
            self._labels_buffer.extend(label_ids)
            self._positions_buffer.extend(range(len(input_ids)))
            self._sample_idx += 1
            if len(self._inputs_buffer) == self.seq_len:
                yield self._flush_buffers()
        while True:
            for sample in self._get_data_iter():
                # pyrefly: ignore [bad-argument-type]
                result = self._tokenize_sample(sample)
                if result is None:
                    self._sample_idx += 1
                    continue

                input_ids, label_ids = result
                remaining = self.seq_len - len(self._inputs_buffer)

                if len(input_ids) > remaining and len(self._inputs_buffer) > 0:
                    pad_len = remaining
                    self._inputs_buffer.extend([self._eos_id] * pad_len)
                    self._labels_buffer.extend([IGNORE_INDEX] * pad_len)
                    self._positions_buffer.extend(range(pad_len))
                    self._pending_input_ids = input_ids
                    self._pending_label_ids = label_ids
                    yield self._flush_buffers()
                    input_ids = self._pending_input_ids
                    label_ids = self._pending_label_ids
                    self._pending_input_ids = []
                    self._pending_label_ids = []

                self._inputs_buffer.extend(input_ids)
                self._labels_buffer.extend(label_ids)
                self._positions_buffer.extend(range(len(input_ids)))
                self._sample_idx += 1

                if len(self._inputs_buffer) == self.seq_len:
                    yield self._flush_buffers()

            # Flush remaining buffer at end of data
            if len(self._inputs_buffer) > 0:
                pad_len = self.seq_len - len(self._inputs_buffer)
                if pad_len > 0:
                    self._inputs_buffer.extend([self._eos_id] * pad_len)
                    self._labels_buffer.extend([IGNORE_INDEX] * pad_len)
                    self._positions_buffer.extend(range(pad_len))

                yield self._flush_buffers()

            if not self._reloop_or_exhaust():
                break

    def _flush_buffers(self):
        """Convert buffers to tensors, clear them, and return the batch."""
        input_tensor = torch.tensor(self._inputs_buffer, dtype=torch.long)
        label_tensor = torch.tensor(self._labels_buffer, dtype=torch.long)
        positions_tensor = torch.tensor(self._positions_buffer, dtype=torch.long)
        self._inputs_buffer = []
        self._labels_buffer = []
        self._positions_buffer = []
        return {"input": input_tensor, "positions": positions_tensor}, label_tensor

    def _state_extras(self) -> dict[str, Any]:
        return {
            "inputs_buffer": self._inputs_buffer,
            "labels_buffer": self._labels_buffer,
            "positions_buffer": self._positions_buffer,
            "pending_input_ids": self._pending_input_ids,
            "pending_label_ids": self._pending_label_ids,
        }

    def _load_state_extras(self, state_dict: dict[str, Any]) -> None:
        self._inputs_buffer = state_dict["inputs_buffer"]
        self._labels_buffer = state_dict["labels_buffer"]
        self._positions_buffer = state_dict["positions_buffer"]
        self._pending_input_ids = state_dict["pending_input_ids"]
        self._pending_label_ids = state_dict["pending_label_ids"]


class _ChatDatasetMixin:
    """Shared _build_dataset for single- and multi-source chat loaders."""

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
        dataset = load_dataset(source.dataset_path, **source.load_dataset_kwargs)
        return ChatDataset(
            dataset=dataset,  # pyrefly: ignore [bad-argument-type]
            tokenizer=tokenizer,
            sample_processor=source.sample_processor,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=source.infinite,
        )


class ChatDataLoader(_ChatDatasetMixin, HFDataLoader):
    """Chat dataloader for instruction/conversation datasets."""

    @dataclass(kw_only=True, slots=True)
    class Config(HFDataLoader.Config):
        dataset_path: str | None = None
        """HuggingFace dataset path (e.g., 'openai/gsm8k') or local path. Required."""

        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
        """Extra kwargs passed to datasets.load_dataset()."""

        sample_processor: Annotated[Callable, tyro.conf.Suppress]
        """Callable(sample_dict) -> list[message_dict]. Set in config functions."""

        def __post_init__(self) -> None:
            if not self.dataset_path:
                raise ValueError(
                    "Config requires dataset_path to be set "
                    "(e.g., 'openai/gsm8k' or 'json')."
                )


@dataclass(kw_only=True, slots=True)
class ChatDataSource(ChatDataLoader.Config):
    """Represent one chat dataset source and its sampling weight"""

    weight: float = 1
    """Data Source sampling weight"""


class InterleavedChatDataLoader(_ChatDatasetMixin, InterleavedHFDataLoader):
    """Configurable chat dataloader that wraps multiple ChatDataset."""

    @dataclass(kw_only=True, slots=True)
    class Config(InterleavedHFDataLoader.Config):
        sources: list[ChatDataSource] = field(default_factory=list)
        """List of datasources to interleave"""
