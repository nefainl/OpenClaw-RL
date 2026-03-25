from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RewardSignalKind = Literal["binary", "directional", "combined"]
SuggestedRLMethod = Literal["binary", "opd", "combined"]
ConsentScope = Literal["local_only", "hive_anonymous", "hive_attributed"]


class RewardSignal(BaseModel):
    """
    PR10A reward signal (from `openclaw` research events).

    This matches `src/research/events/types.ts`:
    - kind ∈ {binary,directional,combined}
    - source ∈ {user_explicit,user_implicit,env_outcome,approval_decision}
    - confidence ∈ [0,1]
    - scalar ∈ [-1,1] (optional)
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    kind: RewardSignalKind
    source: Literal["user_explicit", "user_implicit", "env_outcome", "approval_decision"]
    confidence: float = Field(ge=0.0, le=1.0)
    scalar: float | None = Field(default=None, ge=-1.0, le=1.0)
    hintText: str | None = None


class TrajectoryTurn(BaseModel):
    """
    One line inside PR10A:
      `rl-feed/trajectories/{packageId}.jsonl`

    Key alignment:
    - `turnId`, `contentHash`, `stepIdx` are always present.
    - `contentScrubbed`, `toolName`, `rewardSignal` are optional.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    turnId: str
    role: Literal["user", "assistant", "tool"]
    contentHash: str
    contentScrubbed: str | None = None
    toolName: str | None = None
    rewardSignal: RewardSignal | None = None
    stepIdx: int


class RewardsFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    packageId: str
    signals: list[RewardSignal]


class MetadataFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schemaVersion: Literal["trajectory.v2"]
    packageId: str
    agentId: str
    createdAt: int
    runId: str
    sessionId: str
    dominantSignalKind: RewardSignalKind
    suggestedRLMethod: SuggestedRLMethod
    skillsActivated: list[str]
    sessionRecallHits: int
    scrubbed: bool
    consentScope: ConsentScope
    turnCount: int


@dataclass(frozen=True)
class ParsedPackage:
    package_id: str
    turns: list[TrajectoryTurn]
    rewards: RewardsFile
    metadata: MetadataFile


def parse_jsonl_turns(jsonl_text: str) -> list[TrajectoryTurn]:
    """
    Parse PR10A trajectories JSONL into validated `TrajectoryTurn` objects.

    Fail fast on:
    - invalid JSON
    - strict field/key mismatches
    """

    turns: list[TrajectoryTurn] = []
    for line_no, line in enumerate(jsonl_text.splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise ValueError(f"Invalid JSON in trajectories at line {line_no}: {exc}") from exc
        turns.append(TrajectoryTurn.model_validate(payload))
    return turns

