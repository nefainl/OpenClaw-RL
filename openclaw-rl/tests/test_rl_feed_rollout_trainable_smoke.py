from __future__ import annotations

import sys
from pathlib import Path

OPENCLAW_RL_APP_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(OPENCLAW_RL_APP_DIR))

from rl_feed.data_source import RlFeedDataSource  # noqa: E402
from rl_feed.mapper import Sample  # noqa: E402
from rl_feed.rollout import generate_rollout_rl_feed  # noqa: E402


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "rl-feed" / "v1"


def _install_dummy_slime_processing_utils(monkeypatch) -> None:
    import types

    slime_mod = types.ModuleType("slime")
    slime_utils_mod = types.ModuleType("slime.utils")
    processing_mod = types.ModuleType("slime.utils.processing_utils")

    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            # Contract validation: v1 mapper excludes role=tool turns from prompt tokenization.
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert not tool_msgs, "did not expect role=tool messages in prompt_messages"
            text = "".join(str(m.get("content", "")) for m in messages)
            if add_generation_prompt:
                text += "<gen>"
            if tokenize:
                return [ord(ch) % 50 for ch in text]
            return text

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(ch) % 50 for ch in str(text)]}

    def load_tokenizer(name_or_path: str, trust_remote_code: bool = True):
        return DummyTokenizer()

    processing_mod.load_tokenizer = load_tokenizer

    monkeypatch.setitem(sys.modules, "slime", slime_mod)
    monkeypatch.setitem(sys.modules, "slime.utils", slime_utils_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", processing_mod)


class _Args:
    n_samples_per_prompt = 1
    rollout_batch_size = 2
    hf_checkpoint = "dummy"

    def __init__(self, *, prompt_data: str, save: str | None) -> None:
        self.prompt_data = prompt_data
        self.save = save


def test_offline_rollout_produces_trainable_samples_for_plaintext_package(monkeypatch, tmp_path: Path) -> None:
    _install_dummy_slime_processing_utils(monkeypatch)

    args = _Args(prompt_data=str(FIXTURE_ROOT), save=str(tmp_path))
    ds = RlFeedDataSource(args)
    out = generate_rollout_rl_feed(args, rollout_id=0, data_buffer=ds, evaluation=False)

    assert len(out) == 2  # pkg-0001 then pkg-0002

    s_hash = out[0][0]
    assert s_hash.metadata["packageId"] == "pkg-0001"
    assert s_hash.tokens == []
    assert s_hash.remove_sample is True
    assert s_hash.status == Sample.Status.FAILED

    s_plain = out[1][0]
    assert s_plain.metadata["packageId"] == "pkg-0002"
    assert s_plain.remove_sample is False
    assert s_plain.status == Sample.Status.COMPLETED
    assert s_plain.tokens
    assert s_plain.loss_mask is not None
    assert s_plain.response_length == len(s_plain.loss_mask)

