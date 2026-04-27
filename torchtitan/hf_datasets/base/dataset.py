# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast

from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger


class HFDatasetBase(IterableDataset, Stateful):
    """Shared plumbing for HuggingFace-backed iterable datasets.

    Owns:
      - distributed splitting via ``split_dataset_by_node``
      - resume-aware iteration over map- and iterable-style HF datasets
      - ``_sample_idx`` / ``_epoch`` bookkeeping
      - checkpointing of the underlying HF dataset
      - re-loop behavior (re-shuffle map-style; ``set_epoch`` iterable-style)

    Subclasses must implement:
      - ``__iter__``: tokenization + packing loop. Should increment
        ``self._sample_idx`` per consumed sample, and call
        ``self._reloop_or_exhaust()`` at the end of an iteration pass
        over the dataset to either continue or break.
      - ``_state_extras()``: dict of per-subclass state to checkpoint
        (buffers, pending tokens, etc.).
      - ``_load_state_extras(sd)``: restore state produced by
        ``_state_extras()``.

    Contract:
      - Base owns ``_sample_idx`` and ``_epoch``.
      - Subclasses own their own buffers; base does not touch them.
      - ``epoch`` is always saved at the top level of ``state_dict()``.
    """

    _RESHUFFLE_SEED_BASE = 42

    def __init__(
        self,
        dataset: Any,  # datasets.Dataset or datasets.IterableDataset
        tokenizer: BaseTokenizer,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        infinite: bool,
        dataset_id: str,
    ) -> None:
        self._original_data = split_dataset_by_node(dataset, dp_rank, dp_world_size)
        self._data = self._original_data
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.dataset_id = dataset_id

        self._sample_idx = 0
        self._epoch = 0

    def _get_data_iter(self):
        """Return an iterator positioned at ``self._sample_idx``.

        For map-style datasets, skips to the correct index.
        For iterable-style datasets, the underlying iterator already
        points to the correct index after ``load_state_dict``.
        """
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            return iter(self._data.skip(self._sample_idx))
        return iter(self._data)

    def _reloop_or_exhaust(self) -> bool:
        """Call at the end of a pass over the dataset.

        Returns ``True`` if iteration should continue — i.e. the dataset
        is infinite and has just been re-looped (advances ``self._epoch``,
        resets ``self._sample_idx``, re-shuffles map-style or calls
        ``set_epoch`` iterable-style, logs the re-loop).

        Returns ``False`` when the dataset is non-infinite and has run
        out of data; the subclass ``__iter__`` should break.
        """
        if not self.infinite:
            logger.warning(f"Dataset '{self.dataset_id}' has run out of data")
            return False

        self._sample_idx = 0
        self._epoch += 1
        if isinstance(self._data, Dataset):
            self._data = cast(
                Dataset,
                self._original_data.shuffle(
                    seed=self._RESHUFFLE_SEED_BASE + self._epoch
                ),
            )
        elif hasattr(self._data, "set_epoch"):
            self._data.set_epoch(self._epoch)

        logger.warning(
            f"Dataset '{self.dataset_id}' is being re-looped (epoch {self._epoch})"
        )
        return True

    def state_dict(self) -> dict[str, Any]:
        sd: dict[str, Any] = {
            "epoch": self._epoch,
            **self._state_extras(),
        }
        if isinstance(self._data, Dataset):
            sd["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to efficiently resume from it.
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            sd["data"] = self._data.state_dict()
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._epoch = state_dict.get("epoch", 0)
        self._load_state_extras(state_dict)

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
            # Replay the same per-epoch shuffle so _data matches the order
            # observed at checkpoint time. Epoch 0 stays unshuffled, which
            # preserves bit-identical resume for single-epoch training runs.
            if self._epoch > 0:
                self._data = cast(
                    Dataset,
                    self._original_data.shuffle(
                        seed=self._RESHUFFLE_SEED_BASE + self._epoch
                    ),
                )
        else:
            assert "data" in state_dict
            data_state = state_dict["data"]
            # HuggingFace IterableDataset sync epoch
            saved_epoch = data_state.get("epoch", 0)
            self._data.set_epoch(saved_epoch)
            self._data.load_state_dict(data_state)

    def __iter__(self):
        raise NotImplementedError

    def _state_extras(self) -> dict[str, Any]:
        """Return subclass-specific state (buffers, pending tokens, etc.)."""
        raise NotImplementedError

    def _load_state_extras(self, state_dict: dict[str, Any]) -> None:
        """Restore subclass-specific state from ``state_dict``."""
        raise NotImplementedError
