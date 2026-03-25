from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .mapper import package_to_sample_groups
from .reader import discover_packages, read_package

logger = logging.getLogger(__name__)


def _maybe_load_processed_set(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if isinstance(payload, dict) and isinstance(payload.get("processed"), list):
        return {str(x) for x in payload["processed"]}
    if isinstance(payload, list):
        return {str(x) for x in payload}
    return set()


def _persist_processed_set(path: Path, processed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {"processed": sorted(processed)}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class RlFeedDataSource:
    """
    Offline data source for SLIME/OpenClaw-RL training.

    - `args.prompt_data` is treated as the `rl-feed/` root directory.
    - If `args.save` is provided, we keep a `processed.json` file to avoid
      re-ingesting the same packageIds within the same training run.
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        root = getattr(args, "prompt_data", None)
        if not root:
            raise ValueError("RlFeedDataSource requires args.prompt_data pointing at an rl-feed root directory")
        self.root = Path(root)

        save_dir = getattr(args, "save", None)
        self.processed_path: Path | None = None
        if save_dir:
            self.processed_path = Path(save_dir) / "processed.json"

        discovered = discover_packages(self.root)
        self._processed: set[str] = _maybe_load_processed_set(self.processed_path) if self.processed_path else set()
        self._remaining: list[str] = [pid for pid in discovered if pid not in self._processed]
        self._cursor = 0

        logger.info(
            "[RlFeedDataSource] root=%s discovered=%d remaining=%d processed_file=%s",
            self.root,
            len(discovered),
            len(self._remaining),
            str(self.processed_path) if self.processed_path else "none",
        )

    def get_samples(self, num_samples: int) -> list[list[Any]]:
        if num_samples <= 0:
            return []

        if self._cursor >= len(self._remaining):
            return []

        take = self._remaining[self._cursor : self._cursor + num_samples]
        self._cursor += len(take)

        groups: list[list[Any]] = []
        for pid in take:
            pkg = read_package(self.root, pid)
            groups.extend(package_to_sample_groups(self.args, pkg))
            if pid not in self._processed:
                self._processed.add(pid)

        if self.processed_path and take:
            _persist_processed_set(self.processed_path, self._processed)

        return groups

    def add_samples(self, samples: list[list[Any]]):
        raise RuntimeError(f"{self.__class__.__name__} is read-only")

    def save(self, rollout_id: int):
        # Idempotency persists via processed.json in this v1 implementation.
        return

    def load(self, rollout_id: int | None = None):
        # Reload processed.json if present.
        if self.processed_path:
            self._processed = _maybe_load_processed_set(self.processed_path)
            self._remaining = [pid for pid in discover_packages(self.root) if pid not in self._processed]
            self._cursor = 0

