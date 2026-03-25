from __future__ import annotations

import sys
from pathlib import Path

import pytest

OPENCLAW_RL_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(OPENCLAW_RL_APP_DIR))

from rl_feed.data_source import RlFeedDataSource  # noqa: E402
from rl_feed.rollout import generate_rollout_rl_feed  # noqa: E402


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "rl-feed" / "v1"


class _Args:
    n_samples_per_prompt = 1
    rollout_batch_size = 1

    def __init__(self, *, prompt_data: str, save: str | None) -> None:
        self.prompt_data = prompt_data
        self.save = save


def test_offline_rollout_returns_non_empty_groups(tmp_path: Path) -> None:
    args = _Args(prompt_data=str(FIXTURE_ROOT), save=str(tmp_path))
    ds = RlFeedDataSource(args)
    out = generate_rollout_rl_feed(args, rollout_id=0, data_buffer=ds, evaluation=False)

    assert isinstance(out, list)
    assert out, "expected at least one sample group"
    assert len(out[0]) == _Args.n_samples_per_prompt

