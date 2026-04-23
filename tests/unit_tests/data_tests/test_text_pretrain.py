# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for HuggingFaceTextDataset and its single- and multi-source loaders."""

import unittest

from datasets import load_dataset

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text import (
    DATASETS,
    HFDataSource,
    HuggingFaceTextDataLoader,
    HuggingFaceTextDataset,
    InterleavedHuggingFaceTextDataLoader,
)

from ._helpers import (
    DummyTokenizer,
    TOKENIZER_PATH,
    assert_dataloader_resumes,
)


class TestHuggingFaceTextDataset(unittest.TestCase):
    """Dataset-level behaviors: positions, re-loop shuffling."""

    def test_positions_reset_after_eos_and_bos(self):
        """Per-document positions reset to 0 at EOS/BOS boundaries and
        otherwise increment by 1."""
        tokenizer = DummyTokenizer()
        seq_len = 512

        dl_config = HuggingFaceTextDataLoader.Config(
            dataset="c4_test", num_workers=0, infinite=False
        )
        dl = HuggingFaceTextDataLoader(
            dl_config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=8,
        )

        for batch, _ in zip(map(lambda x: x[0], dl), range(10)):
            input_ids = batch["input"]
            positions = batch["positions"]
            for ids_row, pos_row in zip(input_ids, positions):
                for i, (tok, pos) in enumerate(zip(ids_row, pos_row)):
                    self.assertLess(pos.item(), seq_len)
                    self.assertGreaterEqual(pos.item(), 0)
                    if i == 0:
                        self.assertEqual(pos.item(), 0)
                    if i > 0 and pos.item() > 0:
                        self.assertEqual(pos.item(), pos_row[i - 1].item() + 1)
                    if tok == tokenizer.eos_id and i < len(ids_row) - 1:
                        self.assertEqual(pos_row[i + 1].item(), 0)
                    if tok == tokenizer.bos_id and i > 0:
                        self.assertEqual(pos.item(), 0)

    def test_map_style_shuffle_on_reloop(self):
        """Re-looping a map-style (``Dataset``) source should change order
        every epoch (https://github.com/pytorch/torchtitan/issues/2733).

        Validates without draining a full epoch:
          1. After an epoch boundary, ``_data`` is a shuffled copy of
             ``_original_data`` — not the same object.
          2. ``state_dict()`` carries ``epoch`` so resume knows the seed.
          3. ``load_state_dict()`` replays the same ``shuffle(seed=42+epoch)``.
          4. Legacy checkpoints without ``epoch`` still load (default to 0).
        """

        def _build_ds():
            return HuggingFaceTextDataset(
                dataset_name="c4_test",
                dataset_path=None,
                tokenizer=HuggingFaceTokenizer.Config().build(
                    tokenizer_path=TOKENIZER_PATH
                ),
                seq_len=128,
                dp_rank=0,
                dp_world_size=1,
                infinite=True,
            )

        # 1) Trigger re-loop by fast-forwarding to end
        ds = _build_ds()
        original = ds._data
        ds._sample_idx = len(ds._data)
        ds._epoch = 0
        next(iter(ds))
        self.assertEqual(ds._epoch, 1)
        self.assertIsNot(ds._data, original)

        # 2) state_dict persists epoch
        state = ds.state_dict()
        self.assertEqual(state.get("epoch"), 1)

        # 3) load_state_dict replays shuffle
        ds_resumed = _build_ds()
        ds_resumed.load_state_dict(state)
        self.assertEqual(ds_resumed._epoch, 1)
        self.assertIsNot(ds_resumed._data, ds_resumed._original_data)
        self.assertEqual(
            list(ds._data[:5]["text"]), list(ds_resumed._data[:5]["text"])
        )

        # 4) legacy checkpoint back-compat
        legacy_state = {
            "inputs_buffer": [],
            "positions_buffer": [],
            "sample_idx": 0,
        }
        ds_legacy = _build_ds()
        ds_legacy.load_state_dict(legacy_state)
        self.assertEqual(ds_legacy._epoch, 0)
        self.assertIs(ds_legacy._data, ds_legacy._original_data)


class TestHuggingFaceTextDataLoader(unittest.TestCase):
    """Loader construction / config plumbing."""

    def test_local_batch_size_and_num_workers_plumbed(self):
        dl_config = HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
            num_workers=2,
        )
        dl = HuggingFaceTextDataLoader(
            dl_config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=DummyTokenizer(),
            seq_len=512,
            local_batch_size=8,
        )
        self.assertEqual(dl.batch_size, 8)
        self.assertEqual(dl.num_workers, 2)


class TestHuggingFaceTextCheckpointing(unittest.TestCase):
    """state_dict / load_state_dict round-trip across dp ranks and both
    map-style and streaming datasets."""

    def setUp(self):
        DATASETS["c4_test_streaming"] = DatasetConfig(
            path="tests/assets/c4_test",
            loader=lambda path: load_dataset(path, split="train").to_iterable_dataset(
                num_shards=4
            ),
            sample_processor=lambda sample: sample["text"],
        )

    def tearDown(self):
        del DATASETS["c4_test_streaming"]

    def _build_loader(self, dataset_name, batch_size, seq_len, world_size, rank):
        tokenizer_config = HuggingFaceTokenizer.Config()
        dl_config = HuggingFaceTextDataLoader.Config(
            dataset=dataset_name, infinite=True
        )
        return dl_config.build(
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=tokenizer_config.build(tokenizer_path=TOKENIZER_PATH),
            seq_len=seq_len,
            local_batch_size=batch_size,
        )

    def test_c4_resumption(self):
        for dataset_name in ["c4_test", "c4_test_streaming"]:
            for world_size in [2, 4]:
                for rank in range(world_size):
                    with self.subTest(
                        dataset=dataset_name, world_size=world_size, rank=rank
                    ):
                        assert_dataloader_resumes(
                            self,
                            lambda: self._build_loader(
                                dataset_name, 1, 1024, world_size, rank
                            ),
                            warmup=2050,
                            verify=500,
                        )


class TestInterleavedHuggingFaceTextDataLoader(unittest.TestCase):
    """Behavior of the interleaved text loader.

    Config validation (empty sources, mixed infinite) is tested once in
    ``test_interleave.py`` since it lives on the base class.
    """

    def _make_config(self, **kwargs) -> InterleavedHuggingFaceTextDataLoader.Config:
        defaults = dict(
            sources=[
                HFDataSource(dataset="c4_test", weight=1.0, infinite=False),
                HFDataSource(dataset="c4_test", weight=1.0, infinite=False),
            ],
            seed=42,
            num_workers=0,
        )
        defaults.update(kwargs)
        return InterleavedHuggingFaceTextDataLoader.Config(**defaults)

    def test_local_batch_size_and_num_workers_plumbed(self):
        config = self._make_config(num_workers=2)
        dl = InterleavedHuggingFaceTextDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=DummyTokenizer(),
            seq_len=512,
            local_batch_size=4,
        )
        self.assertEqual(dl.batch_size, 4)
        self.assertEqual(dl.num_workers, 2)

    def test_yields_input_and_positions_keys(self):
        config = self._make_config()
        dl = InterleavedHuggingFaceTextDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=DummyTokenizer(),
            seq_len=512,
            local_batch_size=2,
        )
        batch_input, _ = next(iter(dl))
        self.assertIn("input", batch_input)
        self.assertIn("positions", batch_input)
        self.assertEqual(batch_input["input"].shape[0], 2)
        self.assertEqual(batch_input["input"].shape[1], 512)

    def test_single_source_equivalent_to_single_loader(self):
        """A single-source interleaved loader matches the shape produced
        by the single-source loader with the same config."""
        tokenizer = DummyTokenizer()
        seq_len = 512
        local_batch_size = 4

        single_dl = HuggingFaceTextDataLoader(
            HuggingFaceTextDataLoader.Config(
                dataset="c4_test", num_workers=0, infinite=False
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
        )
        interleaved_dl = InterleavedHuggingFaceTextDataLoader(
            self._make_config(
                sources=[HFDataSource(dataset="c4_test", weight=1.0, infinite=False)],
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
        )

        single_batch_input, _ = next(iter(single_dl))
        interleaved_batch_input, _ = next(iter(interleaved_dl))
        self.assertEqual(
            single_batch_input["input"].shape,
            interleaved_batch_input["input"].shape,
        )
        self.assertEqual(
            single_batch_input["positions"].shape,
            interleaved_batch_input["positions"].shape,
        )


class TestInterleavedHuggingFaceTextCheckpointing(unittest.TestCase):
    def setUp(self):
        DATASETS["c4_test_streaming"] = DatasetConfig(
            path="tests/assets/c4_test",
            loader=lambda path: load_dataset(path, split="train").to_iterable_dataset(
                num_shards=4
            ),
            sample_processor=lambda sample: sample["text"],
        )

    def tearDown(self):
        del DATASETS["c4_test_streaming"]

    def _build_loader(self, batch_size, seq_len, world_size, rank):
        tokenizer_config = HuggingFaceTokenizer.Config()
        dl_config = InterleavedHuggingFaceTextDataLoader.Config(
            sources=[
                HFDataSource(dataset="c4_test", weight=1.0, infinite=True),
                HFDataSource(dataset="c4_test_streaming", weight=2.0, infinite=True),
            ],
            seed=42,
            num_workers=0,
        )
        return dl_config.build(
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=tokenizer_config.build(tokenizer_path=TOKENIZER_PATH),
            seq_len=seq_len,
            local_batch_size=batch_size,
        )

    def test_resumption_across_reloop(self):
        for world_size in [2, 4]:
            for rank in range(world_size):
                with self.subTest(world_size=world_size, rank=rank):
                    assert_dataloader_resumes(
                        self,
                        lambda: self._build_loader(1, 1024, world_size, rank),
                        warmup=2050,
                        verify=500,
                    )

    def test_resumption_mid_epoch(self):
        assert_dataloader_resumes(
            self,
            lambda: self._build_loader(1, 512, world_size=1, rank=0),
            warmup=10,
            verify=50,
        )


if __name__ == "__main__":
    unittest.main()
