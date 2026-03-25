from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrajectoryTurn(BaseModel):
    """
    Minimal representation of one turn inside PR10A `trajectories/{packageId}.jsonl`.

    PR10A is documented as hashes + roles (privacy-first), so this model is
    written to be "hash-compatible" (no plaintext required for parsing).
    """

    model_config = ConfigDict(extra="ignore", strict=True)

    idx: int | None = None
    role: str

    # Hash-only export fields (privacy-preserving). If plaintext export is added
    # later, these fields may be replaced or supplemented.
    promptHash: str | None = None
    responseHash: str | None = None

    # Optional plaintext export fields (not expected in v1; mapper will refuse
    # training if plaintext is missing).
    prompt_text: str | None = None
    response_text: str | None = None

    @model_validator(mode="after")
    def _require_role_and_hash_or_plaintext(self) -> "TrajectoryTurn":
        if not self.role or not isinstance(self.role, str):
            raise ValueError("TrajectoryTurn.role must be a non-empty string")

        have_hash = bool(self.promptHash) or bool(self.responseHash)
        have_plaintext = bool(self.prompt_text) or bool(self.response_text)
        if not (have_hash or have_plaintext):
            raise ValueError(
                "TrajectoryTurn must include at least one of: promptHash/responseHash or prompt_text/response_text"
            )
        return self


class RewardSignal(BaseModel):
    """
    Minimal reward signal shape for PR10A `rewards/{packageId}.json`.
    """

    model_config = ConfigDict(extra="ignore", strict=True)

    kind: str | None = None

    # OpenClaw-RL online path uses `{"score": ...}` in Sample.reward.
    # For offline feed ingestion, support both `score` and `scalar`.
    score: float | None = None
    scalar: float | None = None

    # Optional fields for provenance / confidence.
    source: str | None = None
    confidence: float | None = None
    hintText: str | None = None

    @model_validator(mode="after")
    def _require_score_like(self) -> "RewardSignal":
        if self.score is None and self.scalar is None:
            raise ValueError("RewardSignal must provide either `score` or `scalar`")
        return self

    def to_score(self) -> float:
        return float(self.score if self.score is not None else self.scalar)  # type: ignore[arg-type]


class RewardsFile(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    packageId: str
    signals: list[RewardSignal] = Field(default_factory=list)


class MetadataFile(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    packageId: str
    dominantSignalKind: str | None = None
    suggestedRLMethod: str | None = None

    # Keep other fields possible, but v1 mapper only relies on these.


@dataclass(frozen=True)
class ParsedPackage:
    package_id: str
    turns: list[TrajectoryTurn]
    rewards: RewardsFile
    metadata: MetadataFile


def parse_jsonl_turns(jsonl_text: str) -> list[TrajectoryTurn]:
    """
    Parse PR10A trajectories JSONL into validated TrajectoryTurn objects.
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

