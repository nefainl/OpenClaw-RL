from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

# Make `rl_feed` importable when running from the planning repo root.
OPENCLAW_RL_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(OPENCLAW_RL_APP_DIR))

from rl_feed.reader import discover_packages, read_package  # noqa: E402


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "rl-feed" / "v1"
PKG_ID_1 = "pkg-0001"
PKG_ID_2 = "pkg-0002"


def test_discover_packages_is_deterministic() -> None:
    package_ids = discover_packages(FIXTURE_ROOT)
    assert package_ids == [PKG_ID_1, PKG_ID_2]


def test_read_package_loads_and_validates() -> None:
    pkg = read_package(FIXTURE_ROOT, PKG_ID_1)
    assert pkg.package_id == PKG_ID_1
    assert len(pkg.turns) == 2
    assert pkg.rewards.packageId == PKG_ID_1
    assert pkg.metadata.packageId == PKG_ID_1
    assert pkg.metadata.dominantSignalKind == "binary"


def test_read_package_plaintext_turns_loads() -> None:
    pkg = read_package(FIXTURE_ROOT, PKG_ID_2)
    assert pkg.package_id == PKG_ID_2
    assert pkg.turns[0].prompt_text is not None
    assert pkg.turns[1].response_text is not None


def test_read_package_rejects_missing_file(tmp_path: Path) -> None:
    # Copy the fixture tree and remove one required file.
    dst = tmp_path / "rl-feed"
    shutil.copytree(FIXTURE_ROOT, dst)
    (dst / "metadata" / f"{PKG_ID_1}.meta.json").unlink()

    with pytest.raises(FileNotFoundError):
        _ = read_package(dst, PKG_ID_1)


def test_read_package_rejects_package_id_mismatch(tmp_path: Path) -> None:
    dst = tmp_path / "rl-feed"
    shutil.copytree(FIXTURE_ROOT, dst)

    meta_path = dst / "metadata" / f"{PKG_ID_1}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["packageId"] = "pkg-mismatch"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError):
        _ = read_package(dst, PKG_ID_1)

