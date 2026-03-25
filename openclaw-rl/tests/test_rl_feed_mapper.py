from __future__ import annotations

import sys
from pathlib import Path

import pytest

OPENCLAW_RL_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(OPENCLAW_RL_APP_DIR))

from rl_feed.mapper import Sample, package_to_sample_groups  # noqa: E402
from rl_feed.reader import read_package  # noqa: E402


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "rl-feed" / "v1"
PKG_ID_1 = "pkg-0001"
PKG_ID_2 = "pkg-0002"


class _Args:
    n_samples_per_prompt = 1
    # Intentionally no hf_checkpoint: hash-only path must not try tokenization.


def _sig(sample) -> dict:
    return {
        "group_index": sample.group_index,
        "index": sample.index,
        "tokens": sample.tokens,
        "response_length": sample.response_length,
        "reward": sample.reward,
        "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
        "remove_sample": sample.remove_sample,
        "loss_mask": sample.loss_mask,
    }


def test_mapper_maps_reward_and_refuses_training_when_hash_only() -> None:
    pkg = read_package(FIXTURE_ROOT, PKG_ID_1)
    groups = package_to_sample_groups(_Args(), pkg)

    assert len(groups) == 1
    assert len(groups[0]) == _Args.n_samples_per_prompt

    s = groups[0][0]
    assert s.reward == {"score": 1.0}
    assert s.tokens == []
    assert s.response_length == 0
    assert s.remove_sample is True
    assert s.status == Sample.Status.FAILED
    assert s.loss_mask is None


def test_mapper_is_deterministic() -> None:
    pkg = read_package(FIXTURE_ROOT, PKG_ID_1)
    out1 = package_to_sample_groups(_Args(), pkg)
    out2 = package_to_sample_groups(_Args(), pkg)
    assert _sig(out1[0][0]) == _sig(out2[0][0])


def _install_dummy_slime_processing_utils(monkeypatch) -> None:
    """
    Unit tests run without heavy ML deps (torch/transformers).
    We inject a dummy `slime.utils.processing_utils` module so the mapper's
    lazy import of `load_tokenizer` succeeds.
    """
    import types

    slime_mod = types.ModuleType("slime")
    slime_utils_mod = types.ModuleType("slime.utils")
    processing_mod = types.ModuleType("slime.utils.processing_utils")

    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            # Contract validation: v2 tokenization requires role=tool turns.
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert tool_msgs, "expected role=tool turns in prompt_messages"
            for tm in tool_msgs:
                assert tm.get("content", None) is not None, "tool turns require contentScrubbed"

            def render(m):
                role = m.get("role")
                content = str(m.get("content", ""))
                if role == "user":
                    return f"<|user|>{content}"
                if role == "tool":
                    return f"<|tool|>{content}"
                if role == "assistant":
                    return f"<|assistant|>{content}<|/assistant|>"
                return f"<|{role}|>{content}"

            text = "".join(render(m) for m in messages)
            if add_generation_prompt:
                # Prompt-only call appends the assistant generation prefix (no content yet).
                text += "<|assistant|>"

            if tokenize:
                return [ord(ch) % 50 for ch in text]
            return text

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(ch) % 50 for ch in str(text)]}

    def load_tokenizer(name_or_path: str, trust_remote_code: bool = True):
        return DummyTokenizer()

    processing_mod.load_tokenizer = load_tokenizer

    monkeypatch.setitem(__import__("sys").modules, "slime", slime_mod)
    monkeypatch.setitem(__import__("sys").modules, "slime.utils", slime_utils_mod)
    monkeypatch.setitem(__import__("sys").modules, "slime.utils.processing_utils", processing_mod)


class _ArgsPlaintext:
    n_samples_per_prompt = 1
    hf_checkpoint = "dummy"


def test_mapper_produces_trainable_tokens_loss_mask_when_plaintext_available(monkeypatch) -> None:
    _install_dummy_slime_processing_utils(monkeypatch)

    pkg = read_package(FIXTURE_ROOT, PKG_ID_2)
    groups = package_to_sample_groups(_ArgsPlaintext(), pkg)
    s = groups[0][0]

    # Plaintext tokenization path should succeed.
    assert s.remove_sample is False
    assert s.status == Sample.Status.COMPLETED
    assert isinstance(s.prompt, str) and s.prompt.endswith("<|assistant|>")
    assert isinstance(s.response, str) and s.response.startswith("Hello from assistant")
    assert s.response.endswith("<|/assistant|>")
    assert s.tokens, "expected non-empty tokens when plaintext export is present"
    assert s.loss_mask is not None
    assert s.response_length == len(s.loss_mask)
    assert all(v == 1 for v in s.loss_mask)

    # Deterministic mapping: reward is still mapped to OpenClaw-RL convention.
    assert s.reward == {"score": 1.0}

