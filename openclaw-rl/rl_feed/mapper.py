from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .types import ParsedPackage, RewardSignal, TrajectoryTurn


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


def _select_reward_score(signals: Iterable[RewardSignal], dominant_kind: str | None) -> float:
    signals_list = list(signals)
    if not signals_list:
        raise ValueError("RewardsFile.signals is empty; cannot derive scalar reward score")

    if dominant_kind:
        for s in signals_list:
            if s.kind == dominant_kind:
                return s.to_score()

    # Fallback: first signal with a score/scalar.
    return signals_list[0].to_score()


def _maybe_tokenize_from_plaintext(args: Any, turns: list[TrajectoryTurn]) -> tuple[list[int], list[int], str]:
    """
    Tokenize plaintext prompt/response into (tokens, loss_mask, response_text).

    This is only attempted if plaintext fields are present in the feed export.
    In this repo's planning context, PR10A is hashes-only, so tests won't hit
    this path.
    """

    # Lazy import to avoid pulling transformers in environments that only run unit tests.
    from slime.utils.processing_utils import load_tokenizer  # type: ignore

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    # Heuristic: use prompt_text from all turns that have it; use the last
    # available response_text as the training response.
    prompt_messages: list[dict[str, str]] = []
    last_response: str | None = None
    for t in turns:
        if t.prompt_text:
            prompt_messages.append({"role": t.role, "content": t.prompt_text})
        if t.response_text:
            last_response = t.response_text

    if not prompt_messages or not last_response:
        raise ValueError("Plaintext export detected but missing prompt_text/response_text for tokenization")

    # Mirror openclaw_api_server's "apply_chat_template(messages, tokenize=True, add_generation_prompt=True)" workflow.
    input_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    response_ids = tokenizer(last_response, add_special_tokens=False)["input_ids"]

    # Tokens = prompt + response. Loss is response-only.
    tokens = list(input_ids) + list(response_ids)
    loss_mask = [1] * len(response_ids)
    return tokens, loss_mask, last_response


def package_to_sample_groups(args: Any, package: ParsedPackage) -> list[list[Any]]:
    """
    Convert a package into SLIME sample groups.

    Chosen v1 policy:
    - One package => one group.
    - If token reconstruction is not possible (hash-only export), emit samples
      with reward set, but mark them FAILED and `remove_sample=True` so
      training pipelines can safely drop them.
    """

    dominant_kind = getattr(package.metadata, "dominantSignalKind", None)
    score = _select_reward_score(package.rewards.signals, dominant_kind)
    reward = {"score": float(score)}

    n_samples_per_prompt = int(getattr(args, "n_samples_per_prompt", 1))

    # PR10A turns (or future plaintext-enabled exports) are expected to carry
    # prompt text and response text on different turns. Tokenization is only
    # possible if we have at least one prompt and one response.
    has_prompt_text = any(t.prompt_text is not None for t in package.turns)
    has_response_text = any(t.response_text is not None for t in package.turns)
    plaintext_ok = has_prompt_text and has_response_text

    group_index = 0
    base_index = 0

    tokens: list[int] = []
    response_length = 0
    response_text = ""
    loss_mask: list[int] | None = None
    status = Sample.Status.FAILED if hasattr(Sample, "Status") else "failed"  # type: ignore[comparison-overlap]
    remove_sample = True

    if plaintext_ok:
        # Tokenization is best-effort. If it fails, keep refusal mode.
        try:
            tokens, loss_mask, response_text = _maybe_tokenize_from_plaintext(args, package.turns)
            response_length = len(tokens) - 0  # response_length should be response-only, corrected below
            # loss_mask is response-only.
            response_length = len(loss_mask)
            status = Sample.Status.COMPLETED  # type: ignore[assignment]
            remove_sample = False
        except Exception:  # pragma: no cover
            # Hash-only contract is the expected path; keep samples non-trainable on errors.
            tokens = []
            response_length = 0
            response_text = ""
            loss_mask = None
            status = Sample.Status.FAILED  # type: ignore[assignment]
            remove_sample = True

    group: list[Any] = []
    for i in range(n_samples_per_prompt):
        s = Sample()
        s.group_index = group_index
        s.index = base_index + i
        s.tokens = list(tokens)
        s.response = response_text
        s.response_length = int(response_length)
        s.reward = reward
        s.loss_mask = loss_mask
        s.status = status
        s.remove_sample = remove_sample
        s.metadata = {"packageId": package.package_id}
        group.append(s)

    return [group]

