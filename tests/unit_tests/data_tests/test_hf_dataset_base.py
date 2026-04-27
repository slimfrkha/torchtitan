# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for HFDatasetBase (torchtitan/hf_datasets/base/dataset.py).

Covers _reloop_or_exhaust, _get_data_iter, and state_dict / load_state_dict
for both map-style (datasets.Dataset) and iterable-style (IterableDataset)
backends.  The concrete end-to-end checkpoint behaviour of the text and
multimodal loaders is tested separately in test_text_pretrain.py et al.
"""

import unittest
from typing import Any

from datasets import Dataset, IterableDataset

from torchtitan.hf_datasets.base.dataset import HFDatasetBase

from ._helpers import DummyTokenizer


class _ConcreteDataset(HFDatasetBase):
    """Minimal concrete subclass used only to exercise HFDatasetBase internals."""

    def __iter__(self):
        for sample in self._get_data_iter():
            self._sample_idx += 1
            yield {"text": sample["text"]}, 0

    def _state_extras(self) -> dict[str, Any]:
        return {"extra_key": "extra_val"}

    def _load_state_extras(self, state_dict: dict[str, Any]) -> None:
        pass


def _make_map_ds(n: int = 20, infinite: bool = False) -> _ConcreteDataset:
    raw = Dataset.from_dict({"text": [f"sample_{i}" for i in range(n)]})
    return _ConcreteDataset(
        dataset=raw,
        tokenizer=DummyTokenizer(),
        seq_len=128,
        dp_rank=0,
        dp_world_size=1,
        infinite=infinite,
        dataset_id="test_map",
    )


def _make_iterable_ds(n: int = 20, infinite: bool = False) -> _ConcreteDataset:
    def gen():
        for i in range(n):
            yield {"text": f"sample_{i}"}

    raw = IterableDataset.from_generator(gen)
    return _ConcreteDataset(
        dataset=raw,
        tokenizer=DummyTokenizer(),
        seq_len=128,
        dp_rank=0,
        dp_world_size=1,
        infinite=infinite,
        dataset_id="test_iterable",
    )


class TestReloopOrExhaust(unittest.TestCase):
    def test_finite_returns_false(self):
        ds = _make_map_ds(infinite=False)
        result = ds._reloop_or_exhaust()
        self.assertFalse(result)
        self.assertEqual(ds._epoch, 0)

    def test_infinite_map_style_reshuffles(self):
        ds = _make_map_ds(infinite=True)
        original_data = ds._data
        result = ds._reloop_or_exhaust()
        self.assertTrue(result)
        self.assertEqual(ds._epoch, 1)
        self.assertIsNot(ds._data, original_data)

    def test_infinite_iterable_calls_set_epoch(self):
        ds = _make_iterable_ds(infinite=True)
        if not hasattr(ds._data, "set_epoch"):
            self.skipTest("IterableDataset.set_epoch not available in this HF version")

        called: list[int] = []
        original_fn = ds._data.set_epoch

        def tracking_set_epoch(epoch: int) -> None:
            called.append(epoch)
            original_fn(epoch)

        ds._data.set_epoch = tracking_set_epoch

        result = ds._reloop_or_exhaust()
        self.assertTrue(result)
        self.assertEqual(ds._epoch, 1)
        self.assertEqual(called, [1])


class TestGetDataIter(unittest.TestCase):
    def test_map_style_at_end_returns_empty(self):
        ds = _make_map_ds(n=5)
        ds._sample_idx = len(ds._data)
        self.assertEqual(list(ds._get_data_iter()), [])

    def test_map_style_skips_to_sample_idx(self):
        ds = _make_map_ds(n=10)
        ds._sample_idx = 3
        first = next(ds._get_data_iter())
        self.assertEqual(first["text"], "sample_3")

    def test_iterable_style_returns_iter(self):
        ds = _make_iterable_ds(n=5)
        first = next(ds._get_data_iter())
        self.assertIn("text", first)
        self.assertEqual(first["text"], "sample_0")


class TestStateDict(unittest.TestCase):
    def test_map_style_round_trip(self):
        ds = _make_map_ds(n=20)
        ds._sample_idx = 7
        sd = ds.state_dict()

        self.assertEqual(sd["epoch"], 0)
        self.assertEqual(sd["sample_idx"], 7)
        self.assertIn("extra_key", sd)
        self.assertNotIn("data", sd)

        ds2 = _make_map_ds(n=20)
        ds2.load_state_dict(sd)
        self.assertEqual(ds2._epoch, 0)
        self.assertEqual(ds2._sample_idx, 7)

    def test_map_style_reshuffled_epoch_preserved(self):
        ds = _make_map_ds(n=20, infinite=True)
        ds._reloop_or_exhaust()  # epoch → 1, data reshuffled
        ds._sample_idx = 4
        sd = ds.state_dict()

        self.assertEqual(sd["epoch"], 1)
        self.assertEqual(sd["sample_idx"], 4)

        ds2 = _make_map_ds(n=20, infinite=True)
        ds2.load_state_dict(sd)
        self.assertEqual(ds2._epoch, 1)
        self.assertEqual(ds2._sample_idx, 4)
        # load_state_dict must replay the same shuffle so both see the same order
        self.assertIsNot(ds2._data, ds2._original_data)
        self.assertEqual(
            list(ds._data[:5]["text"]),
            list(ds2._data[:5]["text"]),
        )

    def test_load_state_dict_defaults_epoch_zero(self):
        """Legacy checkpoints without 'epoch' key must load with epoch=0."""
        ds = _make_map_ds(n=10)
        legacy_sd = {"sample_idx": 3, "extra_key": "extra_val"}
        ds.load_state_dict(legacy_sd)
        self.assertEqual(ds._epoch, 0)
        self.assertEqual(ds._sample_idx, 3)

    def test_iterable_state_dict_uses_data_key(self):
        """Iterable datasets must be checkpointed under 'data', not 'sample_idx'."""
        ds = _make_iterable_ds(n=10)
        sd = ds.state_dict()
        self.assertIn("data", sd)
        self.assertNotIn("sample_idx", sd)
        self.assertEqual(sd["epoch"], 0)


if __name__ == "__main__":
    unittest.main()
