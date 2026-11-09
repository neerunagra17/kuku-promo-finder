"""On-disk run state, so an interrupted or failed run resumes where it stopped.

<out>/state/
  manifest.json          provider, pinned model, prompt version, fingerprint of the pack
  moments/ep_007.json    one file per finished episode, written only from a valid model reply
  judgments/<key>.json   one file per judged clip (key = hash of its segments and beats)
  usage.jsonl            one line per API attempt (tokens, seconds, retries) across all reruns
  failures.jsonl         one line per failed request: what failed and why
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_write(path: Path, obj) -> None:
    """Write via a temp file + rename, so a crash never leaves a half-written result."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def pack_fingerprint(episodes) -> str:
    h = hashlib.sha256()
    for e in sorted(episodes, key=lambda e: e.number):
        h.update(json.dumps([e.number, [(ev.id, ev.start, ev.end, ev.text) for ev in e.events]]).encode())
    return h.hexdigest()[:16]


class StateStore:
    def __init__(self, root):
        self.root = Path(root)
        self.moments_dir = self.root / "moments"
        self.judgments_dir = self.root / "judgments"
        self._lock = threading.Lock()

    def init(self):
        self.moments_dir.mkdir(parents=True, exist_ok=True)
        self.judgments_dir.mkdir(parents=True, exist_ok=True)

    def clear(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # manifest
    def manifest(self) -> dict | None:
        p = self.root / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def write_manifest(self, data: dict):
        self.init()
        atomic_write(self.root / "manifest.json", {**data, "updated_at": now()})

    # moments, one file per episode
    def _ep(self, n: int) -> Path:
        return self.moments_dir / f"ep_{n:03d}.json"

    def has_moments(self, n: int) -> bool:
        return self._ep(n).exists()

    def load_moments(self, n: int) -> dict:
        return json.loads(self._ep(n).read_text(encoding="utf-8"))

    def save_moments(self, n: int, record: dict):
        atomic_write(self._ep(n), {**record, "saved_at": now()})

    # judgments, one file per clip
    def _j(self, key: str) -> Path:
        return self.judgments_dir / f"{key}.json"

    def has_judgment(self, key: str) -> bool:
        return self._j(key).exists()

    def load_judgment(self, key: str) -> dict:
        return json.loads(self._j(key).read_text(encoding="utf-8"))

    def save_judgment(self, key: str, record: dict):
        atomic_write(self._j(key), {**record, "saved_at": now()})

    # append-only logs
    def _append(self, name: str, row: dict):
        with self._lock:
            self.init()
            with open(self.root / name, "a", encoding="utf-8") as f:
                f.write(json.dumps({**row, "at": now()}, ensure_ascii=False) + "\n")

    def _rows(self, name: str) -> list[dict]:
        p = self.root / name
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    def log_usage(self, **row):
        self._append("usage.jsonl", row)

    def log_failure(self, **row):
        self._append("failures.jsonl", row)

    def usage_totals(self) -> dict:
        t = {"calls": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        for row in self._rows("usage.jsonl"):
            for k in t:
                t[k] += row.get(k, 0) or 0
        t["seconds"] = round(t["seconds"], 1)
        return t

    def failure_count(self) -> int:
        return len(self._rows("failures.jsonl"))
