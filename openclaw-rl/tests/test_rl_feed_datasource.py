from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

OPENCLAW_RL_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(OPENCLAW_RL_APP_DIR))

from rl_feed.data_source import RlFeedDataSource  # noqa: E402


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "rl-feed" / "v1"
PKG_ID_1 = "pkg-0001"
PKG_ID_2 = "pkg-0002"


class _Args:
    n_samples_per_prompt = 1

    def __init__(self, *, prompt_data: str, save: str | None) -> None:
        self.prompt_data = prompt_data
        self.save = save


def test_datasource_is_idempotent_via_processed_json(tmp_path: Path) -> None:
    args = _Args(prompt_data=str(FIXTURE_ROOT), save=str(tmp_path))
    ds = RlFeedDataSource(args)

    groups1 = ds.get_samples(1)
    assert len(groups1) == 1
    assert groups1[0][0].metadata["packageId"] == PKG_ID_1

    processed_path = tmp_path / "processed.json"
    assert processed_path.is_file()
    processed = json.loads(processed_path.read_text(encoding="utf-8"))
    assert processed["processed"] == [PKG_ID_1]

    groups2 = ds.get_samples(1)
    assert len(groups2) == 1
    assert groups2[0][0].metadata["packageId"] == PKG_ID_2

    processed = json.loads(processed_path.read_text(encoding="utf-8"))
    assert processed["processed"] == [PKG_ID_1, PKG_ID_2]


def test_datasource_returns_empty_when_no_remaining_packages(tmp_path: Path) -> None:
    # After ingesting all fixture packages, the datasource should return no more groups.
    args = _Args(prompt_data=str(FIXTURE_ROOT), save=str(tmp_path))
    ds = RlFeedDataSource(args)
    _ = ds.get_samples(1)
    _ = ds.get_samples(1)
    assert ds.get_samples(1) == []

