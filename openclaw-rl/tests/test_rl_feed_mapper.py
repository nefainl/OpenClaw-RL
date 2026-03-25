from __future__ import annotations

import sys
from pathlib import Path

import pytest

OPENCLAW_RL_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(OPENCLAW_RL_APP_DIR))

from rl_feed.mapper import Sample, package_to_sample_groups  # noqa: E402
from rl_feed.reader import read_package  # noqa: E402


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "rl-feed" / "v1"
PKG_ID = "pkg-0001"


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
    pkg = read_package(FIXTURE_ROOT, PKG_ID)
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
    pkg = read_package(FIXTURE_ROOT, PKG_ID)
    out1 = package_to_sample_groups(_Args(), pkg)
    out2 = package_to_sample_groups(_Args(), pkg)
    assert _sig(out1[0][0]) == _sig(out2[0][0])

