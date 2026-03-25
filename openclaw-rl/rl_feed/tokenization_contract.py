from __future__ import annotations

from typing import Any, Sequence

from .types import TrajectoryTurn


class TokenizationRefusedError(ValueError):
    """Raised when we cannot safely reconstruct prompt/response tokens."""


def _require_content_scrubbed(turn: TrajectoryTurn, *, role: str, context: str) -> str:
    content = turn.contentScrubbed
    if content is None:
        raise TokenizationRefusedError(f"Missing contentScrubbed for {role}: {context}")
    return content


def _build_prompt_messages_and_assistant_message(
    turns_sorted: Sequence[TrajectoryTurn],
    *,
    assistant_turn_index: int,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """
    Build message lists for prefix tokenization.

    Contract:
    - prompt_messages includes every turn prior to the target assistant turn.
    - assistant_message is the target assistant turn.
    - contentScrubbed is required for every role in the prompt window,
      including role="tool".
    """

    assistant_turn = turns_sorted[assistant_turn_index]
    if assistant_turn.role != "assistant":
        raise ValueError("assistant_turn_index must point at an assistant turn")

    assistant_content = _require_content_scrubbed(
        assistant_turn,
        role="assistant",
        context=f"assistant_turn_index={assistant_turn_index}",
    )
    assistant_message = {"role": "assistant", "content": assistant_content}

    prompt_messages: list[dict[str, str]] = []
    for idx, t in enumerate(turns_sorted[:assistant_turn_index]):
        if t.role not in ("user", "assistant", "tool"):
            continue

        content = _require_content_scrubbed(
            t,
            role=t.role,
            context=f"turn_index={idx} stepIdx={t.stepIdx} turnId={t.turnId}",
        )
        prompt_messages.append({"role": t.role, "content": content})

    if not prompt_messages:
        # This is a refusal: without any prompt window, tokenization is underspecified.
        raise TokenizationRefusedError("Tokenization refused: prompt_messages is empty")

    return prompt_messages, assistant_message


def _prefix_tokenize_with_contract(
    args: Any,
    *,
    prompt_messages: list[dict[str, str]],
    assistant_message: dict[str, str],
) -> tuple[str, str, list[int], list[int], int]:
    """
    Tokenize according to openclaw_api_server.py prefix contract.

    Normalized semantics:
    - prompt_text = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
    - full_text = tokenizer.apply_chat_template(prompt_messages + [assistant_message], add_generation_prompt=False)
    - response_text = full_text[len(prompt_text):] when full_text starts with prompt_text else full_text
    - prompt_ids = tokenizer(prompt_text)
    - response_ids = tokenizer(response_text)
    - tokens = prompt_ids + response_ids
    - loss_mask = [1] * len(response_ids) (response-only loss)
    - response_length = len(response_ids)
    """

    # Lazy import so unit tests can monkeypatch `slime.utils.processing_utils.load_tokenizer`.
    try:
        from slime.utils.processing_utils import load_tokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        # In minimal unit-test environments, `slime` may not be installed.
        raise TokenizationRefusedError("Missing slime tokenizer dependencies; refusing tokenization") from exc

    if not hasattr(args, "hf_checkpoint"):
        # Offline ingestion can still safely refuse rather than crash.
        raise TokenizationRefusedError("hf_checkpoint missing on args; cannot tokenize")

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + [assistant_message],
        tokenize=False,
        add_generation_prompt=False,
    )

    if isinstance(prompt_text, str) and isinstance(full_text, str) and full_text.startswith(prompt_text):
        response_text = full_text[len(prompt_text) :]
    else:
        # Mirror upstream fallback.
        response_text = full_text

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    tokens = list(prompt_ids) + list(response_ids)
    loss_mask = [1] * len(response_ids)
    response_length = len(response_ids)

    return str(prompt_text), str(response_text), tokens, loss_mask, response_length


def tokenization_contract_tokenize_assistant_turn(
    args: Any,
    *,
    turns_sorted: Sequence[TrajectoryTurn],
    assistant_turn_index: int,
) -> tuple[str, str, list[int], list[int], int]:
    """
    Return (prompt_text, response_text, tokens, loss_mask, response_length) or raise.

    Refusal semantics are represented by TokenizationRefusedError.
    """

    prompt_messages, assistant_message = _build_prompt_messages_and_assistant_message(
        turns_sorted,
        assistant_turn_index=assistant_turn_index,
    )

    return _prefix_tokenize_with_contract(
        args,
        prompt_messages=prompt_messages,
        assistant_message=assistant_message,
    )

