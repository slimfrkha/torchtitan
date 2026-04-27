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
from torchtitan.hf_datasets.text.pretrain import _validate_dataset

from ._helpers import (
    DummyTokenizer,
    TOKENIZER_PATH,
    assert_dataloader_resumes,
)


class TestHuggingFaceTextDataset(unittest.TestCase):
    """Dataset-level behaviors: positions."""

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


class TestHuggingFaceTextDatasetCheckpointing(unittest.TestCase):
    """state_dict / load_state_dict for HuggingFaceTextDataset directly.

    Loader-level resumption is tested in TestHuggingFaceTextCheckpointing.
    """

    def _build_ds(self):
        return HuggingFaceTextDataset(
            dataset_name="c4_test",
            dataset_path=None,
            tokenizer=DummyTokenizer(),
            seq_len=128,
            dp_rank=0,
            dp_world_size=1,
            infinite=False,
        )

    def test_state_dict_round_trip(self):
        ds = self._build_ds()
        it = iter(ds)
        next(it)
        state = ds.state_dict()

        for key in ("epoch", "sample_idx", "inputs_buffer", "positions_buffer"):
            self.assertIn(key, state)
        self.assertGreater(state["sample_idx"], 0)
        self.assertEqual(state["epoch"], 0)

        ds2 = self._build_ds()
        ds2.load_state_dict(state)
        self.assertEqual(ds2._sample_idx, state["sample_idx"])
        self.assertEqual(ds2._epoch, state["epoch"])
        self.assertEqual(ds2._inputs_buffer, state["inputs_buffer"])
        self.assertEqual(ds2._positions_buffer, state["positions_buffer"])

    def test_legacy_checkpoint_missing_positions_buffer(self):
        """Checkpoints without 'positions_buffer' fall back to [] with a warning."""
        ds = self._build_ds()
        legacy_state = {"epoch": 0, "sample_idx": 0, "inputs_buffer": [1, 2, 3]}
        ds.load_state_dict(legacy_state)
        self.assertEqual(ds._inputs_buffer, [1, 2, 3])
        self.assertEqual(ds._positions_buffer, [])


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


class TestValidateDataset(unittest.TestCase):
    def test_known_dataset_returns_path_and_callables(self):
        path, loader, processor = _validate_dataset("c4_test")
        self.assertEqual(path, "tests/assets/c4_test")
        self.assertTrue(callable(loader))
        self.assertTrue(callable(processor))

    def test_unknown_dataset_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_dataset("not_a_real_dataset")
        self.assertIn("not_a_real_dataset", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
