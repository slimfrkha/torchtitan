# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ChatDataset and its single- and multi-source loaders."""

import unittest

from datasets import Dataset

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text import (
    ChatDataLoader,
    ChatDataset,
    ChatDataSource,
    InterleavedChatDataLoader,
)

from ._helpers import (
    SFT_DATA_PATH,
    TOKENIZER_PATH,
    assert_dataloader_resumes,
)

def process_chat_sample(sample):
    """Convert a ``{"question", "answer"}`` sample into [user, assistant] messages."""
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _load_tokenizer():
    return HuggingFaceTokenizer(tokenizer_path=TOKENIZER_PATH)


def _load_dataset():
    return Dataset.from_json(SFT_DATA_PATH)


def _make_chat_dataset(seq_len=2048, infinite=False, sample_processor=None):
    return ChatDataset(
        dataset=_load_dataset(),
        tokenizer=_load_tokenizer(),
        sample_processor=sample_processor or process_chat_sample,
        seq_len=seq_len,
        infinite=infinite,
    )


class TestChatDataset(unittest.TestCase):
    """Dataset-level behaviors: masking, shifting, packing, validation, looping."""

    def test_prompt_masked_response_unmasked(self):
        chat_ds = _make_chat_dataset(seq_len=2048)
        batch, labels = next(iter(chat_ds))
        input_ids = batch["input"]

        self.assertEqual(input_ids.shape, labels.shape)
        self.assertEqual(input_ids.shape[0], 2048)

        masked = (labels == IGNORE_INDEX).nonzero(as_tuple=True)[0]
        unmasked = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        self.assertGreater(len(masked), 0)
        self.assertGreater(len(unmasked), 0)
        self.assertGreater(unmasked[0].item(), 0, "First token label should be masked")

    def test_shifted_by_one(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=process_chat_sample,
            seq_len=2048,
            infinite=False,
        )

        batch, labels = next(iter(chat_ds))
        input_ids = batch["input"]

        sample = ds[0]
        messages = process_chat_sample(sample)
        full_text = tokenizer.apply_chat_template(messages)
        full_tokens = tokenizer.encode(full_text, add_bos=True, add_eos=False)

        expected_input = full_tokens[:-1]
        expected_label = full_tokens[1:]

        seq_len_actual = len(expected_input)
        self.assertEqual(input_ids[:seq_len_actual].tolist(), expected_input)

        prompt_text = tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        response_start = len(prompt_tokens) - 1
        self.assertGreaterEqual(response_start, 0)
        self.assertNotEqual(labels[response_start].item(), IGNORE_INDEX)
        self.assertEqual(
            labels[response_start:seq_len_actual].tolist(),
            expected_label[response_start:],
        )

    def test_packing_multiple_samples_into_one_sequence(self):
        """Small seq_len packs multiple short samples into one batch."""
        seq_len = 256
        chat_ds = _make_chat_dataset(seq_len=seq_len)

        batches = list(chat_ds)
        # 10 samples of ~79-123 effective tokens → fewer than 10 packed batches
        self.assertGreater(len(batches), 0)
        self.assertLess(len(batches), 10)
        for batch, labels in batches:
            self.assertEqual(batch["input"].shape[0], seq_len)
            self.assertEqual(labels.shape[0], seq_len)
            self.assertIn("positions", batch)
            self.assertEqual(batch["positions"].shape[0], seq_len)

    def test_positions_reset_at_document_boundaries(self):
        seq_len = 256
        chat_ds = _make_chat_dataset(seq_len=seq_len)
        batch, _ = next(iter(chat_ds))
        positions = batch["positions"]

        self.assertEqual(positions[0].item(), 0)
        resets = (positions[1:] == 0).nonzero(as_tuple=True)[0]
        self.assertGreater(len(resets), 0)

        pos_list = positions.tolist()
        for i in range(1, len(pos_list)):
            if pos_list[i] == 0:
                continue
            self.assertEqual(pos_list[i], pos_list[i - 1] + 1)

    def test_samples_exceeding_seq_len_are_dropped(self):
        chat_ds = _make_chat_dataset(seq_len=32)
        self.assertEqual(list(chat_ds), [])

    def _run_with_bad_processor(self, bad_processor):
        ds = Dataset.from_list([{"question": "hi", "answer": "bye"}])
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=_load_tokenizer(),
            sample_processor=bad_processor,
            seq_len=2048,
            infinite=False,
        )
        next(iter(chat_ds))

    def test_rejects_wrong_first_role(self):
        with self.assertRaises(ValueError):
            self._run_with_bad_processor(
                lambda s: [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "assistant", "content": "OK"},
                ]
            )

    def test_rejects_wrong_second_role(self):
        with self.assertRaises(ValueError):
            self._run_with_bad_processor(
                lambda s: [
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": "hello again"},
                ]
            )

    def test_rejects_multi_turn(self):
        with self.assertRaises(ValueError):
            self._run_with_bad_processor(
                lambda s: [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "bye"},
                ]
            )

    def test_infinite_unpacked(self):
        chat_ds = _make_chat_dataset(seq_len=2048, infinite=True)
        it = iter(chat_ds)
        samples = [next(it) for _ in range(15)]
        self.assertEqual(len(samples), 15)
        self.assertGreaterEqual(chat_ds._epoch, 1)

    def test_infinite_packed(self):
        chat_ds = _make_chat_dataset(seq_len=256, infinite=True)
        it = iter(chat_ds)
        batches = [next(it) for _ in range(20)]
        self.assertEqual(len(batches), 20)
        self.assertGreaterEqual(chat_ds._epoch, 1)


class TestChatDatasetCheckpointing(unittest.TestCase):
    def test_state_dict_round_trip(self):
        chat_ds = _make_chat_dataset(seq_len=128)
        it = iter(chat_ds)
        next(it)
        state = chat_ds.state_dict()

        for key in (
            "sample_idx", "epoch",
            "inputs_buffer", "labels_buffer", "positions_buffer",
            "pending_input_ids", "pending_label_ids",
        ):
            self.assertIn(key, state)
        self.assertGreater(state["sample_idx"], 0)
        self.assertEqual(state["epoch"], 0)

        chat_ds_resumed = _make_chat_dataset(seq_len=128)
        chat_ds_resumed.load_state_dict(state)
        self.assertEqual(chat_ds_resumed._sample_idx, state["sample_idx"])
        self.assertEqual(chat_ds_resumed._epoch, state["epoch"])
        self.assertEqual(chat_ds_resumed._inputs_buffer, state["inputs_buffer"])
        self.assertEqual(chat_ds_resumed._labels_buffer, state["labels_buffer"])
        self.assertEqual(chat_ds_resumed._positions_buffer, state["positions_buffer"])
        self.assertEqual(chat_ds_resumed._pending_input_ids, state["pending_input_ids"])
        self.assertEqual(chat_ds_resumed._pending_label_ids, state["pending_label_ids"])

        remaining = list(chat_ds_resumed)
        self.assertGreater(len(remaining), 0)

    def test_dataloader_resumes_multi_epoch(self):
        tokenizer_config = HuggingFaceTokenizer.Config()

        def build(streaming, world_size, rank):
            def _factory():
                dl_config = ChatDataLoader.Config(
                    dataset_path="json",
                    load_dataset_kwargs={
                        "data_files": SFT_DATA_PATH,
                        "split": "train",
                        "streaming": streaming,
                    },
                    sample_processor=process_chat_sample,
                    infinite=True,
                )
                return dl_config.build(
                    dp_world_size=world_size,
                    dp_rank=rank,
                    tokenizer=tokenizer_config.build(tokenizer_path=TOKENIZER_PATH),
                    seq_len=128,
                    local_batch_size=1,
                )
            return _factory

        for streaming in [True, False]:
            for world_size in [2, 4]:
                for rank in range(world_size):
                    with self.subTest(
                        streaming=streaming, world_size=world_size, rank=rank
                    ):
                        assert_dataloader_resumes(
                            self,
                            build(streaming, world_size, rank),
                            warmup=8,
                            verify=3,
                        )


def _make_interleaved_chat_config(**kwargs) -> InterleavedChatDataLoader.Config:
    defaults = dict(
        sources=[
            ChatDataSource(
                dataset_path="json",
                load_dataset_kwargs={"data_files": SFT_DATA_PATH, "split": "train"},
                sample_processor=process_chat_sample,
                weight=1.0,
                infinite=True,
            ),
            ChatDataSource(
                dataset_path="json",
                load_dataset_kwargs={"data_files": SFT_DATA_PATH, "split": "train"},
                sample_processor=process_chat_sample,
                weight=2.0,
                infinite=True,
            ),
        ],
        seed=42,
        num_workers=0,
    )
    defaults.update(kwargs)
    return InterleavedChatDataLoader.Config(**defaults)


def _build_interleaved_chat_loader(config, batch_size=1, seq_len=256):
    return config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=HuggingFaceTokenizer.Config().build(tokenizer_path=TOKENIZER_PATH),
        seq_len=seq_len,
        local_batch_size=batch_size,
    )


class TestInterleavedChatDataLoader(unittest.TestCase):
    """Interleaved chat loader behavior.

    Config validation (empty sources, mixed infinite) lives in
    ``test_interleave.py``.
    """

    def test_local_batch_size_and_num_workers_plumbed(self):
        config = _make_interleaved_chat_config(num_workers=2)
        dl = _build_interleaved_chat_loader(config, batch_size=4)
        self.assertEqual(dl.batch_size, 4)
        self.assertEqual(dl.num_workers, 2)

    def test_yields_input_positions_and_labels(self):
        seq_len = 256
        dl = _build_interleaved_chat_loader(
            _make_interleaved_chat_config(), batch_size=2, seq_len=seq_len
        )
        batch_input, batch_label = next(iter(dl))
        self.assertIn("input", batch_input)
        self.assertIn("positions", batch_input)
        self.assertEqual(batch_input["input"].shape, (2, seq_len))
        self.assertEqual(batch_input["positions"].shape, (2, seq_len))
        self.assertEqual(batch_label.shape, (2, seq_len))


class TestInterleavedChatCheckpointing(unittest.TestCase):
    def test_resumption_mid_epoch(self):
        assert_dataloader_resumes(
            self,
            lambda: _build_interleaved_chat_loader(_make_interleaved_chat_config()),
            warmup=5,
            verify=10,
        )

    def test_resumption_across_reloop(self):
        assert_dataloader_resumes(
            self,
            lambda: _build_interleaved_chat_loader(_make_interleaved_chat_config()),
            warmup=30,
            verify=20,
        )


class TestChatDataLoaderConfig(unittest.TestCase):
    def test_missing_dataset_path_raises_value_error(self):
        """ChatDataLoader.Config must raise ValueError when dataset_path is not set."""
        with self.assertRaises(ValueError) as ctx:
            ChatDataLoader.Config(sample_processor=lambda x: x)
        self.assertIn("dataset_path", str(ctx.exception))

    def test_valid_config_constructed_without_error(self):
        config = ChatDataLoader.Config(
            dataset_path="json",
            sample_processor=lambda x: x,
        )
        self.assertEqual(config.dataset_path, "json")


if __name__ == "__main__":
    unittest.main()
