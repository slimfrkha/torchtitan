# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the multimodal dataloader."""

import unittest

from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader

from ._helpers import TOKENIZER_PATH, assert_dataloader_resumes


_TOKENIZER_CONFIG = MultiModalTokenizer.Config(
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    vision_start_token="<|vision_start|>",
    vision_end_token="<|vision_end|>",
    pad_token="<|endoftext|>",
)
_TOKENIZER = _TOKENIZER_CONFIG.build(tokenizer_path=TOKENIZER_PATH)


def _make_mm_config():
    return MMDataLoader.Config(
        dataset="cc12m-test",
        max_images_per_batch=128,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=784,
        max_pixels=200000,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )


class TestMMDatasetCheckpointing(unittest.TestCase):
    """MMDataLoader state_dict / load_state_dict round-trip across dp ranks."""

    def _build_loader(self, batch_size, seq_len, world_size, rank):
        return _make_mm_config().build(
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=_TOKENIZER,
            seq_len=seq_len,
            local_batch_size=batch_size,
        )

    def test_cc12m_resumption(self):
        for world_size in [1, 2]:
            for rank in range(world_size):
                with self.subTest(world_size=world_size, rank=rank):
                    assert_dataloader_resumes(
                        self,
                        lambda: self._build_loader(1, 4096, world_size, rank),
                        warmup=5,
                        verify=10,
                        extra_keys=("pixel_values", "grid_thw"),
                    )


if __name__ == "__main__":
    unittest.main()
