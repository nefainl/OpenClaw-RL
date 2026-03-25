from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .types import MetadataFile, ParsedPackage, RewardsFile, TrajectoryTurn, parse_jsonl_turns


def discover_packages(root: Path) -> list[str]:
    """
    Discover packageIds under an `rl-feed/` root.

    Expected layout:
      root/
        trajectories/{packageId}.jsonl
        rewards/{packageId}.json
        metadata/{packageId}.meta.json
    """

    trajectories_dir = root / "trajectories"
    if not trajectories_dir.is_dir():
        return []

    package_ids = sorted(p.stem for p in trajectories_dir.glob("*.jsonl") if p.is_file())
    return package_ids


def _require_file(path: Path, *, kind: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {kind} file: {path}")


def read_package(root: Path, package_id: str) -> ParsedPackage:
    trajectories_path = root / "trajectories" / f"{package_id}.jsonl"
    rewards_path = root / "rewards" / f"{package_id}.json"
    metadata_path = root / "metadata" / f"{package_id}.meta.json"

    _require_file(trajectories_path, kind="trajectories")
    _require_file(rewards_path, kind="rewards")
    _require_file(metadata_path, kind="metadata")

    # Read + validate rewards + metadata first so we can verify packageId consistency.
    rewards_payload = json.loads(rewards_path.read_text(encoding="utf-8"))
    rewards = RewardsFile.model_validate(rewards_payload)
    if rewards.packageId != package_id:
        raise ValueError(f"packageId mismatch in rewards: expected={package_id} got={rewards.packageId}")

    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = MetadataFile.model_validate(metadata_payload)
    if metadata.packageId != package_id:
        raise ValueError(
            f"packageId mismatch in metadata: expected={package_id} got={metadata.packageId}"
        )

    # Parse trajectory turns.
    turns_text = trajectories_path.read_text(encoding="utf-8")
    turns: list[TrajectoryTurn] = parse_jsonl_turns(turns_text)

    if not turns:
        raise ValueError(f"Empty trajectories file for packageId={package_id}")

    # PR10A exporter encodes `packageId` inside `turnId` as `${packageId}-t${stepIdx}`.
    turn_id_prefix = f"{package_id}-t"
    if not all(t.turnId.startswith(turn_id_prefix) for t in turns):
        bad = next(t.turnId for t in turns if not t.turnId.startswith(turn_id_prefix))
        raise ValueError(
            f"packageId mismatch in trajectories: expected_prefix={turn_id_prefix} bad_turnId={bad}"
        )

    return ParsedPackage(package_id=package_id, turns=turns, rewards=rewards, metadata=metadata)

