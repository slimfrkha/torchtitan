# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.base.interleave import InterleavedDataset


def _base_dataloader_kwargs(config: ParallelAwareDataloader.Config, local_batch_size: int) -> dict[str, Any]:
    """Shared torch DataLoader kwargs read off a ParallelAwareDataloader.Config."""
    return {
        "num_workers": config.num_workers,
        "persistent_workers": config.persistent_workers,
        "pin_memory": config.pin_memory,
        "prefetch_factor": config.prefetch_factor,
        "batch_size": local_batch_size,
    }


class HFDataLoader(ParallelAwareDataloader):
    """Base class for single-source HF-backed dataloaders.

    Subclasses implement ``_build_dataset`` (required) and optionally
    ``_build_collate_fn``. All dataloader plumbing is handled here.
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

        dl_kwargs = _base_dataloader_kwargs(config, local_batch_size)
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
        source,
        *,
        tokenizer: BaseTokenizer,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        local_batch_size: int,
    ):
        """Build the underlying dataset from a config or a DataSource.

        ``source`` is the loader's ``Config`` for single-source loaders,
        or a ``*DataSource`` entry for interleaved loaders. Both share
        the fields the builder needs, so the same hook serves both.
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
        """Optional collate_fn builder. Takes top-level Config since
        collator settings are loader-scoped, not per-source."""
        return None


class InterleavedHFDataLoader(ParallelAwareDataloader):
    """Base class for interleaved multi-source HF-backed dataloaders.

    Subclasses implement ``_build_dataset`` (required) and optionally
    ``_build_collate_fn``. The ``Config`` subclass must define a
    ``sources`` field; the ``__post_init__`` here validates it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        sources: list = field(default_factory=list)
        """List of data sources to interleave. Subclasses should narrow the
        element type, e.g. ``list[HFDataSource]``."""

        seed: int = 42
        """Interleaving seed"""

        def __post_init__(self) -> None:
            if not self.sources:
                raise ValueError("At least one source should be defined.")
            infinite_values = [source.infinite for source in self.sources]
            if len(set(infinite_values)) > 1:
                raise ValueError(
                    f"All data sources must have the same 'infinite' setting, "
                    f"got: {infinite_values}"
                )

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
        ds = InterleavedDataset(
            datasets=[
                self._build_dataset(
                    source,
                    tokenizer=tokenizer,
                    seq_len=seq_len,
                    dp_rank=dp_rank,
                    dp_world_size=dp_world_size,
                    local_batch_size=local_batch_size,
                )
                for source in config.sources
            ],
            weights=[source.weight for source in config.sources],
            seed=config.seed,
        )

        dl_kwargs = _base_dataloader_kwargs(config, local_batch_size)
        collate_fn = self._build_collate_fn(
            config,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
        )
        if collate_fn is not None:
            dl_kwargs["collate_fn"] = collate_fn

        super().__init__(
            ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dl_kwargs,
        )

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
        """Build a dataset for a single ``*DataSource``. Called once per
        source. Signature matches ``HFDataLoader._build_dataset`` so
        flavors can share a builder across single- and multi-source
        loader subclasses."""
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
