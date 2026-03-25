"""
Offline ingestion of PR10A `rl-feed/` packages into SLIME/OpenClaw-RL.

This module is intentionally lightweight and CI-testable without requiring
GPUs or model downloads.
"""

__all__ = [
    "types",
    "reader",
    "mapper",
    "data_source",
    "rollout",
]

