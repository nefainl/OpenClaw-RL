from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from .types import ParsedPackage, RewardSignal, TrajectoryTurn
from .tokenization_contract import (
    TokenizationRefusedError,
    tokenization_contract_tokenize_assistant_turn,
)


def _load_sample_class():
    """
    Try importing SLIME's Sample without hard-failing when heavy deps (torch)
    are not installed (unit-test environment).
    """

    try:
        from slime.utils.types import Sample as SlimeSample  # type: ignore

        return SlimeSample
    except Exception:  # pragma: no cover
        # Minimal compatibility stub for CI/CPU-only parsing tests.
        @dataclass
        class Sample:  # noqa: D401 - simple compatibility stub
            group_index: int | None = None
            index: int | None = None
            prompt: str | list[dict[str, str]] = ""
            tokens: list[int] = field(default_factory=list)
            multimodal_inputs: dict[str, Any] | None = None
            multimodal_train_inputs: dict[str, Any] | None = None
            response: str = ""
            response_length: int = 0
            label: str | None = None
            reward: float | dict[str, Any] | None = None
            loss_mask: list[int] | None = None
            rollout_log_probs: list[float] | None = None
            rollout_routed_experts: list[list[int]] | None = None
            remove_sample: bool = False

            class Status(Enum):
                PENDING = "pending"
                COMPLETED = "completed"
                TRUNCATED = "truncated"
                ABORTED = "aborted"
                FAILED = "failed"

            status: Status = Status.PENDING
            metadata: dict = field(default_factory=dict)
            train_metadata: dict | None = None

        return Sample


Sample = _load_sample_class()

_STATUS_FAILED = Sample.Status.FAILED if hasattr(Sample, "Status") else "failed"  # type: ignore[comparison-overlap]
_STATUS_COMPLETED = Sample.Status.COMPLETED if hasattr(Sample, "Status") else "completed"  # type: ignore[comparison-overlap]


def _map_reward_signal_to_score(signal: RewardSignal) -> float:
    # OpenClaw-RL offline reward must be binary-compatible via scalar sign.
    # - scalar > 0 => 1
    # - scalar < 0 => -1
    # - scalar == 0 or missing => 0
    scalar = signal.scalar
    if scalar is None or scalar == 0:
        return 0.0
    if scalar > 0:
        return 1.0
    return -1.0


def _select_reward_score(signals: Iterable[RewardSignal], dominant_kind: str | None) -> float:
    signals_list = list(signals)
    if not signals_list:
        return 0.0

    if dominant_kind:
        for s in signals_list:
            if s.kind == dominant_kind:
                return _map_reward_signal_to_score(s)

    return _map_reward_signal_to_score(signals_list[0])


def _maybe_tokenize_from_scrubbed_content(
    args: Any,
    *,
    prompt_messages: list[dict[str, str]],
    response_text: str,
) -> tuple[list[int], list[int], int]:
    """
    Tokenize (prompt_messages + response_text) into:
    - tokens = prompt_ids + response_ids
    - loss_mask = [1] * len(response_ids)  (response-only loss)
    - response_length = len(response_ids)

    Tokenization mirrors `external/openclaw-rl/openclaw-rl/openclaw_api_server.py`:
    - prompt_ids from tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)
    - response_ids from tokenizer(response_text, add_special_tokens=False)
    """

    # Lazy import to avoid pulling heavy deps in unit test environments.
    from slime.utils.processing_utils import load_tokenizer  # type: ignore

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    tokens = list(prompt_ids) + list(response_ids)
    loss_mask = [1] * len(response_ids)
    return tokens, loss_mask, len(response_ids)


def _build_prompt_messages_for_assistant(
    turns_sorted: Sequence[TrajectoryTurn],
    assistant_turn_index: int,
) -> tuple[list[dict[str, str]], str]:
    assistant_turn = turns_sorted[assistant_turn_index]

    if assistant_turn.role != "assistant":
        raise ValueError("assistant_turn_index must point at an assistant turn")

    if assistant_turn.contentScrubbed is None:
        raise ValueError("Missing contentScrubbed on assistant turn")

    response_text = assistant_turn.contentScrubbed
    prompt_messages: list[dict[str, str]] = []

    for t in turns_sorted[:assistant_turn_index]:
        # Safety: tool turns do not carry enough structure for tool-aware chat templates in v1,
        # so we exclude them from prompt tokenization.
        if t.role == "tool":
            continue

        if t.role not in ("user", "assistant"):
            continue

        if t.contentScrubbed is None:
            raise ValueError(f"Missing contentScrubbed on {t.role} turn")

        prompt_messages.append({"role": t.role, "content": t.contentScrubbed})

    if not prompt_messages:
        raise ValueError("Tokenization refused: prompt_messages is empty")

    return prompt_messages, response_text


def package_to_sample_groups(args: Any, package: ParsedPackage) -> list[list[Any]]:
    """
    Convert a package into SLIME sample groups.

    Chosen v1 policy:
    - One Sample group per assistant turn (Option A).
    - If token reconstruction is not possible (missing `contentScrubbed` for required
      roles), emit samples with reward set, but mark them FAILED and `remove_sample=True`
      so training pipelines can safely drop them.
    """

    dominant_kind = getattr(package.metadata, "dominantSignalKind", None)
    score = _select_reward_score(package.rewards.signals, str(dominant_kind) if dominant_kind else None)
    reward = {"score": float(score)}

    n_samples_per_prompt = int(getattr(args, "n_samples_per_prompt", 1))

    turns_sorted = sorted(package.turns, key=lambda t: (t.stepIdx, t.turnId))
    assistant_indices = [i for i, t in enumerate(turns_sorted) if t.role == "assistant"]

    groups: list[list[Any]] = []
    for assistant_idx in assistant_indices:
        assistant_turn = turns_sorted[assistant_idx]

        tokens: list[int] = []
        response_length = 0
        response_text = ""
        prompt_text = ""
        loss_mask: list[int] | None = None
        status = _STATUS_FAILED
        remove_sample = True

        try:
            prompt_text, response_text, tokens, loss_mask, response_length = tokenization_contract_tokenize_assistant_turn(
                args,
                turns_sorted=turns_sorted,
                assistant_turn_index=assistant_idx,
            )
            status = _STATUS_COMPLETED
            remove_sample = False
        except TokenizationRefusedError:
            # Deterministic refusal: missing required `contentScrubbed` in prompt window.
            tokens = []
            response_length = 0
            response_text = ""
            prompt_text = ""
            loss_mask = None
            status = _STATUS_FAILED
            remove_sample = True
        except Exception:
            # Unexpected errors should not silently turn into malformed samples.
            raise

        group: list[Any] = []
        group_index = int(assistant_turn.stepIdx)
        base_index = group_index * n_samples_per_prompt

        for i in range(n_samples_per_prompt):
            s = Sample()
            s.group_index = group_index
            s.index = base_index + i
            s.tokens = list(tokens)
            s.prompt = prompt_text
            s.response = response_text
            s.response_length = int(response_length)
            s.reward = reward
            s.loss_mask = loss_mask
            s.status = status
            s.remove_sample = remove_sample
            s.metadata = {
                "packageId": package.package_id,
                "turnId": assistant_turn.turnId,
            }
            group.append(s)

        groups.append(group)

    return groups

