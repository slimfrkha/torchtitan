# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal dataset and dataloader for VLM training.

Workflow overview::

    HuggingFace Dataset (streaming)
            │
            ▼
    ┌───────────────────────────────────────────────────────┐
    │  Sample Processor  (per-sample, in Dataset.__iter__)  │
    │                                                       │
    │  1. Parse raw sample (dataset-specific format)        │
    │     e.g. OBELICS interleaved text/images,             │
    │          CC12M text-image pairs                       │
    │                                                       │
    │  2. Process vision: decode image/video bytes,         │
    │     resize to multiples of (patch_size * merge_size), │
    │     normalize with image_mean/std                     │
    │     → pixel_values: list[Tensor(T,H,W,C)]            │
    │                                                       │
    │  3. Process text: insert vision placeholder tokens    │
    │     <|vision_start|><|image_pad|>...<|vision_end|>    │
    │     into text, then tokenize                          │
    │     → input_ids: Tensor(seq_len,)                     │
    │     → labels: same as input_ids, with vision tokens   │
    │       masked to ignore_id (-100)                      │
    └───────────────────────────────────────────────────────┘
            │
            ▼  (optional, if packing_buffer_size > 0)
    ┌───────────────────────────────────────────────────────┐
    │  Sample Packer                                        │
    │  Bin-pack short samples into seq_len-length sequences │
    │  to reduce padding waste                              │
    └───────────────────────────────────────────────────────┘
            │
            ▼  DataLoader batches samples (batch_size)
    ┌───────────────────────────────────────────────────────┐
    │  Collator  (MultiModalCollator)                    │
    │                                                       │
    │  1. collate_images: for each image Tensor(T,H,W,C),  │
    │     reshape into patches (num_patches, patch_dim),    │
    │     pad all images to same num_patches                │
    │     → pixel_values: (N, max_patches, patch_dim)       │
    │     → grid_thw: (N, 3) per-image [T, H', W'] dims    │
    │     (same for videos)                                 │
    │                                                       │
    │  2. collate_text: pad input_ids/labels across batch   │
    │     to seq_len, pad batch to target batch_size        │
    │     → input_ids: (batch_size, seq_len)                │
    │     → labels: (batch_size, seq_len)                   │
    └───────────────────────────────────────────────────────┘
            │
            ▼
    Model receives: {input_ids, pixel_values, grid_thw,
                     pixel_values_videos, grid_thw_videos,
                     special_tokens: dict[str, int]}, labels
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer, BaseTokenizer

from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.base import HFDatasetBase, HFDataLoader
from torchtitan.tools.logging import logger
from .mm_collator import MultiModalCollator
from .utils.image import calculate_vision_tokens, process_image
from .utils.packing import MMSamplePacker
from .utils.text import insert_vision_placeholders


def _process_mm_sample(
    texts: list[str | None],
    images: list[bytes | None],
    tokenizer: MultiModalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    **kwargs,
) -> dict[str, Any] | None:
    """Common processing logic for multimodal samples.

    Args:
        texts: List of strings with None indicating image positions
        images: List of image bytes with None for text positions
        tokenizer: Tokenizer for text processing
        patch_size: Size of image patches
        spatial_merge_size: merge 2D image patches to reduce LLM's sequence length.
            - if 1 (default): no merge, effectively NoOp
            - if 2: 2x2=4 image patches will be reduced to 1 LLM visual token

    Returns:
        Dict with:
            - input_ids: Tensor of token IDs
            - labels: Tensor of label IDs
            - pixel_values: List of processed image tensors

    Example:
        Interleaved format:
        texts = [text1, None, text2, None, text3]
        images = [None, img1, None, img2, None]

        Image-text pair format as a special case of interleaved:
        texts = [None, text]
        images = [image, None]
    """
    if not texts or len(texts) != len(images):
        return None

    processed_images = []
    num_image_tokens = []

    for idx, img in enumerate(images):
        if img is not None:
            # Resize (to multiples of patch_size x merge_size) and normalize images
            processed_img = process_image(
                img,
                patch_size=patch_size,
                merge_size=spatial_merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                image_mean=image_mean,
                image_std=image_std,
            )
            if processed_img is not None:
                num_tokens, _, _ = calculate_vision_tokens(
                    num_frames=1,
                    height=processed_img.shape[1],
                    width=processed_img.shape[2],
                    patch_size=patch_size,
                    spatial_merge_size=spatial_merge_size,
                    temporal_patch_size=1,
                )
                processed_images.append(processed_img)
                num_image_tokens.append(num_tokens)
                texts[idx] = None

    if len(processed_images) != len([_ for _ in images if _ is not None]):
        logger.warning("Cannot process all images for sample. Dropping")
        return None

    # Replace image placeholders (None) with image token sequences
    processed_text = insert_vision_placeholders(
        texts,
        num_image_tokens,
        # pyrefly: ignore [missing-attribute]
        vision_start_token=tokenizer.vision_start_token,
        # pyrefly: ignore [missing-attribute]
        vision_token=tokenizer.image_token,
        # pyrefly: ignore [missing-attribute]
        vision_end_token=tokenizer.vision_end_token,
        # pyrefly: ignore [bad-argument-type]
        eos_token=tokenizer.eos_token,
    )

    tokens = tokenizer.encode(processed_text)
    input_ids = torch.tensor(tokens)
    labels = torch.tensor(tokens)

    special_token_ids = torch.tensor(
        [
            # pyrefly: ignore [missing-attribute]
            tokenizer.vision_start_id,
            # pyrefly: ignore [missing-attribute]
            tokenizer.vision_end_id,
            # pyrefly: ignore [missing-attribute]
            tokenizer.image_id,
            # pyrefly: ignore [missing-attribute]
            tokenizer.video_id,
        ]
    )
    labels = torch.where(torch.isin(labels, special_token_ids), IGNORE_INDEX, labels)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "positions": torch.arange(len(input_ids)),
        "pixel_values": processed_images,
    }


def _process_obelics_sample(
    sample: dict[str, Any],
    tokenizer: MultiModalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    **kwargs,
) -> dict[str, Any] | None:
    """Process a sample from the OBELICS dataset (interleaved text and images)."""
    return _process_mm_sample(
        texts=sample.get("texts", []),
        images=sample.get("images", []),
        tokenizer=tokenizer,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_mean=image_mean,
        image_std=image_std,
    )


def _process_cc12_wd_sample(
    sample: dict[str, Any],
    tokenizer: MultiModalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    **kwargs,
) -> dict[str, Any] | None:
    """Process a sample from the CC12-WD dataset (text-image pairs)."""
    text = sample.get("txt", "")
    image = sample.get("jpg", None)

    texts = [None, text]
    images = [image, None]

    return _process_mm_sample(
        texts=texts,
        images=images,
        tokenizer=tokenizer,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_mean=image_mean,
        image_std=image_std,
    )


MM_DATASETS = {
    "obelics": DatasetConfig(
        path="HuggingFaceM4/OBELICS",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        sample_processor=_process_obelics_sample,
    ),
    "cc12m": DatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        sample_processor=_process_cc12_wd_sample,
    ),
    "cc12m-test": DatasetConfig(
        path="tests/assets/cc12m_test",
        loader=lambda path: load_dataset(
            path, split="train", data_files={"train": "*.tar"}, streaming=True
        ),
        sample_processor=_process_cc12_wd_sample,
    ),
}


def _validate_mm_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path, returning (path, loader, sample_processor)."""
    if dataset_name not in MM_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(MM_DATASETS.keys())}"
        )

    config = MM_DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceMultiModalDataset(HFDatasetBase):
    """HuggingFace multimodal dataset with support for sample packing."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: MultiModalTokenizer,
        batch_size: int,
        seq_len: int,
        patch_size: int,
        temporal_patch_size: int,
        spatial_merge_size: int,
        min_pixels: int,
        max_pixels: int,
        image_mean: tuple[float, ...],
        image_std: tuple[float, ...],
        packing_buffer_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        video_dir: str = "",
        video_fps: float = 2.0,
        video_min_frames: int = 4,
        video_max_frames: int = 768,
        dataset_subset: str = "",
    ) -> None:
        dataset_name = dataset_name.lower()

        path, dataset_loader, self.sample_processor = _validate_mm_dataset(
            dataset_name, dataset_path
        )

        # Pass subset to loaders that accept it
        sig = inspect.signature(dataset_loader)
        if "subset" in sig.parameters and dataset_subset:
            ds = dataset_loader(path, subset=dataset_subset)
        else:
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

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean
        self.image_std = image_std
        self.video_dir = video_dir
        self.video_fps = video_fps
        self.video_min_frames = video_min_frames
        self.video_max_frames = video_max_frames
        self.enable_packing = packing_buffer_size > 0
        if self.enable_packing:
            self.packer = MMSamplePacker(
                max_seq_length=seq_len,
                buffer_size=packing_buffer_size,
                batch_size=batch_size,
            )

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_idx += 1

                processed = self.sample_processor(
                    sample=sample,
                    tokenizer=self._tokenizer,
                    patch_size=self.patch_size,
                    temporal_patch_size=self.temporal_patch_size,
                    spatial_merge_size=self.spatial_merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    image_mean=self.image_mean,
                    image_std=self.image_std,
                    video_dir=self.video_dir,
                    video_fps=self.video_fps,
                    video_min_frames=self.video_min_frames,
                    video_max_frames=self.video_max_frames,
                    seq_len=self.seq_len,
                )
                if processed is None:
                    continue

                if processed["input_ids"].shape[0] > self.seq_len:
                    logger.warning(
                        f"Sample length {processed['input_ids'].shape[0]} > training {self.seq_len=}. Skip"
                    )
                    continue

                if self.enable_packing:
                    self.packer.add_sample(processed)

                    if self.packer.has_batch_ready():
                        batch = self.packer.get_next_batch()
                        if batch:
                            yield from batch
                else:
                    yield processed

            # Flush leftovers in packer when raw samples are exhausted
            if self.enable_packing:
                self.packer.flush()
                while self.packer.has_batch_ready():
                    # pyrefly: ignore [invalid-yield]
                    yield from self.packer.get_next_batch()
                # Drain any remainder that doesn't fill a full batch
                while self.packer.packed_samples:
                    yield self.packer.packed_samples.popleft()

            if not self._reloop_or_exhaust():
                break

    def _state_extras(self) -> dict:
        if not self.enable_packing:
            return {}
        return {
            "packer_state": {
                "sample_buffer": list(self.packer._sample_buffer.values()),
                "packed_samples": list(self.packer.packed_samples),
            }
        }

    def _load_state_extras(self, state_dict: dict) -> None:
        if self.enable_packing and "packer_state" in state_dict:
            packer_state = state_dict["packer_state"]
            self.packer._sample_buffer = {
                i: s for i, s in enumerate(packer_state["sample_buffer"])
            }
            self.packer._next_id = len(packer_state["sample_buffer"])
            self.packer.packed_samples.clear()
            self.packer.packed_samples.extend(packer_state["packed_samples"])

class MMDataLoader(HFDataLoader):
    """Configurable multimodal dataloader for VLM training."""

    @dataclass(kw_only=True, slots=True)
    class Config(HFDataLoader.Config):
        dataset: str = "cc12m-test"
        """Dataset to use"""

        dataset_subset: str = ""
        """Dataset subset/config name."""

        # Batching configs
        packing_buffer_size: int = 0
        """Set to a value >0 to enable sample packing."""

        max_images_per_batch: int
        """Max images per batch to bound vision encoder memory."""

        # Preprocessing configs
        patch_size: int
        """Patch size of the vision encoder."""

        temporal_patch_size: int
        """Temporal patch size for video processing."""

        spatial_merge_size: int
        """Spatially merge visual tokens after encoder. e.g. 2 means 2x2=4 patches merged."""

        min_pixels: int
        """Minimum number of pixels for image resizing."""

        max_pixels: int
        """Maximum number of pixels for image resizing."""

        image_mean: tuple[float, ...]
        """Per-channel mean for image normalization."""

        image_std: tuple[float, ...]
        """Per-channel std for image normalization."""

        video_dir: str = ""
        """Base directory for video files (for datasets with video filename references)."""

        video_fps: float = 2.0
        """Target frames per second for video sampling."""

        video_min_frames: int = 4
        """Minimum number of frames to sample from a video."""

        video_max_frames: int = 768
        """Maximum number of frames to sample from a video."""

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
        assert isinstance(tokenizer, MultiModalTokenizer)
        return HuggingFaceMultiModalDataset(
            dataset_name=source.dataset,
            dataset_path=source.dataset_path,
            tokenizer=tokenizer,
            batch_size=local_batch_size,
            seq_len=seq_len,
            patch_size=source.patch_size,
            temporal_patch_size=source.temporal_patch_size,
            spatial_merge_size=source.spatial_merge_size,
            min_pixels=source.min_pixels,
            max_pixels=source.max_pixels,
            image_mean=source.image_mean,
            image_std=source.image_std,
            packing_buffer_size=source.packing_buffer_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=source.infinite,
            video_dir=source.video_dir,
            video_fps=source.video_fps,
            video_min_frames=source.video_min_frames,
            video_max_frames=source.video_max_frames,
            dataset_subset=source.dataset_subset,
        )

    def _build_collate_fn(
        self,
        config: HFDataLoader.Config,
        *,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
    ):
        assert isinstance(tokenizer, MultiModalTokenizer)
        assert isinstance(config, MMDataLoader.Config)
        return MultiModalCollator(
            batch_size=local_batch_size,
            seq_len=seq_len,
            max_images_per_batch=config.max_images_per_batch,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            tokenizer=tokenizer,
        )
