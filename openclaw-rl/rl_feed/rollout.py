from __future__ import annotations

from typing import Any

__all__ = ["generate_rollout_rl_feed"]


def generate_rollout_rl_feed(args: Any, rollout_id: int, data_buffer: Any, evaluation: bool = False):
    """
    SLIME `--rollout-function-path` entrypoint for offline `rl-feed/` ingestion.

    This rollout does not call any model / generator. It only returns prebuilt
    `Sample` groups from the provided `DataSource`.
    """

    if evaluation:
        # Mirror existing examples: allow evaluation mode only if your training
        # loop expects it, but for this v1 offline adapter we keep the behavior simple.
        raise RuntimeError("generate_rollout_rl_feed (v1) does not support evaluation mode")

    rollout_batch_size = int(getattr(args, "rollout_batch_size", 0))
    if rollout_batch_size <= 0:
        raise ValueError("generate_rollout_rl_feed requires args.rollout_batch_size > 0")

    return data_buffer.get_samples(rollout_batch_size)

