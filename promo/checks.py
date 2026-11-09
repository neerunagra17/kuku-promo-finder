"""Automatic quality checks on every promo. Hard checks gate a promo; soft checks flag it."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import Index, Promo

MIN_LEN, MAX_LEN, HOOK_WINDOW = 20.0, 60.0, 3.0


def _check(ok: bool, level: str, detail: str) -> dict:
    return {"ok": bool(ok), "level": level, "detail": detail}


def run_checks(p: "Promo", idx: "Index") -> dict:
    out = {}
    d = p.duration
    out["duration_20_60s"] = _check(MIN_LEN <= d <= MAX_LEN, "hard", f"{d:.1f}s")

    hook = idx.event(p.hook_event)
    first = p.segments[0]
    lead = hook.start - first.start
    out["hook_in_first_3s"] = _check(hook.episode == first.episode and 0 <= lead <= HOOK_WINDOW,
                                     "hard", f"hook starts at {lead:.1f}s")

    cliff = idx.event(p.cliff_event)
    last = p.segments[-1]
    out["ends_on_cliffhanger"] = _check(cliff.episode == last.episode and
                                        last.start <= cliff.start and cliff.end <= last.end + 0.05,
                                        "hard", f"cut {last.end - cliff.end:+.1f}s after the cliff beat")

    # nothing that answers the cliffhanger may sit between the cliff beat and the cut
    ep = idx.episodes[last.episode]
    after = [e for e in ep.events if e.type == "dialogue" and cliff.end <= e.start < last.end]
    out["no_answer_before_cut"] = _check(not after, "hard",
                                         "ok" if not after else f"{len(after)} line(s) after the cliff beat")

    segs_ok = all(s.duration >= 3.0 for s in p.segments) and len(p.segments) <= 3
    for a, b in zip(p.segments, p.segments[1:]):
        if a.episode == b.episode and b.start < a.end:
            segs_ok = False
    out["segments_valid"] = _check(segs_ok, "hard", f"{len(p.segments)} segment(s)")

    mid = []
    for s in p.segments:
        sep = idx.episodes[s.episode]
        for e in sep.events:
            if e.type != "dialogue":
                continue
            if e.start + 0.2 < s.start < e.end - 0.2:
                mid.append(f"starts inside {e.id}")
            if e.start + 0.2 < s.end < e.end - 0.2:
                mid.append(f"ends inside {e.id}")
    out["no_mid_line_cut"] = _check(not mid, "soft", "; ".join(mid) or "ok")

    # a cut between episodes is fine for the viewer, but the editor should know
    out["single_episode"] = _check(len(p.episodes) == 1, "soft",
                                   f"episodes {', '.join(map(str, p.episodes))}")
    return out
