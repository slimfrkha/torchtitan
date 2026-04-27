# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer


class HFDataLoader(ParallelAwareDataloader):
    """Base class for HF-backed dataloaders.

    Subclasses implement ``_build_dataset`` (required) and optionally
    ``_build_collate_fn``. The returned dataset may be a single
    ``HFDatasetBase`` or an ``InterleavedDataset`` wrapping several —
    that decision belongs to the subclass.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        infinite: bool = True
        """Whether to loop the dataset infinitely"""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        dataset = self._build_dataset(
            config,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            local_batch_size=local_batch_size,
        )

        dl_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
        }
        collate_fn = self._build_collate_fn(
            config,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
        )
        if collate_fn is not None:
            dl_kwargs["collate_fn"] = collate_fn

        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dl_kwargs,
        )

    def _build_dataset(
        self,
        config,
        *,
        tokenizer: BaseTokenizer,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        local_batch_size: int,
    ) -> IterableDataset:
        """Return the iterable to wrap.

        Single-source flavors return one ``HFDatasetBase``; interleaved
        flavors return an ``InterleavedDataset`` of per-source datasets.
        """
        raise NotImplementedError

    def _build_collate_fn(
        self,
        config: Config,
        *,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
    ):
        return None
