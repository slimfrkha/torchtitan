# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``torchtitan.components.dataloader.ParallelAwareDataloader``.

HF-text / HF-chat / MM loader tests live under ``tests/unit_tests/data/``.
"""

import unittest

from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader


class DummyDataset(IterableDataset):
    """Yields 100 ``({input: i}, i)`` samples."""

    def __iter__(self):
        for i in range(100):
            yield {"input": i}, i


class TestParallelAwareDataloader(unittest.TestCase):
    def test_dataloader_yields_correct_batches(self):
        dataset = DummyDataset()
        batch_size = 4

        dataloader = ParallelAwareDataloader(
            dataset,
            dp_rank=0,
            dp_world_size=1,
            batch_size=batch_size,
        )

        batches = list(dataloader)
        self.assertEqual(len(batches), 25)

        first_batch_input, first_batch_label = batches[0]
        self.assertEqual(len(first_batch_input["input"]), batch_size)
        self.assertEqual(len(first_batch_label), batch_size)
        self.assertEqual(first_batch_input["input"].tolist(), [0, 1, 2, 3])
        self.assertEqual(first_batch_label.tolist(), [0, 1, 2, 3])

        last_batch_input, last_batch_label = batches[-1]
        self.assertEqual(last_batch_input["input"].tolist(), [96, 97, 98, 99])
        self.assertEqual(last_batch_label.tolist(), [96, 97, 98, 99])

    def test_validate_kwargs_rejects_invalid_kwargs(self):
        dataset = DummyDataset()
        with self.assertRaises(ValueError) as context:
            ParallelAwareDataloader(
                dataset,
                dp_rank=0,
                dp_world_size=1,
                invalid_arg=42,
            )
        self.assertIn("Invalid dataloader kwargs", str(context.exception))
        self.assertIn("invalid_arg", str(context.exception))

    def test_config_batch_size_overwritten_by_explicit_batch_size(self):
        """Explicit ``batch_size`` kwarg must override the one from config."""
        dataset = DummyDataset()
        config_kwargs = {"batch_size": 2, "num_workers": 0}
        explicit_batch_size = 8

        dataloader_kwargs = {
            **config_kwargs,
            "batch_size": explicit_batch_size,
        }
        dataloader = ParallelAwareDataloader(
            dataset,
            dp_rank=0,
            dp_world_size=1,
            **dataloader_kwargs,
        )
        self.assertEqual(dataloader.batch_size, explicit_batch_size)


if __name__ == "__main__":
    unittest.main()
