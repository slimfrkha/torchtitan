# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared test helpers for data/ unit tests.

Exposes:
  - Path constants for test assets
  - ``DummyTokenizer`` — a minimal ``BaseTokenizer`` for loaders that need a
    tokenizer but not a real one
  - ``process_chat_sample`` — question/answer -> single-turn messages
  - ``assert_resume_matches`` and ``assert_dataloader_resumes`` — factor out
    the "checkpoint, rebuild, verify next N match" pattern used everywhere
"""

import os
from copy import deepcopy

import torch

from torchtitan.components.tokenizer import BaseTokenizer


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
TOKENIZER_PATH = os.path.join(ASSETS_DIR, "tokenizer")
SFT_DATA_PATH = os.path.join(ASSETS_DIR, "sft_test", "data.json")


class DummyTokenizer(BaseTokenizer):
    """Tokenizer that encodes characters as their ASCII value.

    Useful for loader wiring / positions / shape tests where a real
    tokenizer would only slow the test down.
    """

    def __init__(self):
        super().__init__()
        self.eos_id = 2
        self.bos_id = 1

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        tokens = [ord(c) for c in text]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids if t > 2)

    def get_vocab_size(self) -> int:
        return 256


def assert_resume_matches(
    test,
    it_orig,
    it_resumed,
    *,
    verify: int,
    extra_keys: tuple[str, ...] = (),
) -> None:
    """Consume ``verify`` batches from both iterators and assert identity.

    Checks ``input``, ``positions``, and labels by default. ``extra_keys``
    names additional fields in the batch dict (e.g. ``"pixel_values"``,
    ``"grid_thw"``) whose presence/absence and shape are compared but
    values are not (kept loose so image tensors don't need bit-identical
    reproduction across runs).
    """
    for step in range(verify):
        expected_input, expected_labels = next(it_orig)
        input_dict, labels = next(it_resumed)

        test.assertTrue(
            torch.equal(input_dict["input"], expected_input["input"]),
            f"input mismatch at step {step}",
        )
        test.assertTrue(
            torch.equal(input_dict["positions"], expected_input["positions"]),
            f"positions mismatch at step {step}",
        )
        test.assertTrue(
            torch.equal(labels, expected_labels),
            f"labels mismatch at step {step}",
        )

        for key in extra_keys:
            exp = expected_input[key]
            res = input_dict[key]
            test.assertEqual(
                exp is None, res is None, f"{key} None mismatch at step {step}"
            )
            if exp is not None:
                test.assertEqual(
                    exp.shape, res.shape, f"{key} shape mismatch at step {step}"
                )


def assert_dataloader_resumes(
    test,
    build_fn,
    *,
    warmup: int,
    verify: int,
    extra_keys: tuple[str, ...] = (),
) -> None:
    """High-level: build → consume warmup → checkpoint → rebuild → verify match.

    ``build_fn`` is a thunk that returns a fresh dataloader; it is called
    twice so both loaders start from the same initial state.
    """
    dl = build_fn()
    it = iter(dl)
    for _ in range(warmup):
        next(it)
    state = deepcopy(dl.state_dict())

    dl_resumed = build_fn()
    dl_resumed.load_state_dict(state)
    it_resumed = iter(dl_resumed)

    assert_resume_matches(test, it, it_resumed, verify=verify, extra_keys=extra_keys)
