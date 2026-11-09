"""Promo finder pipeline.

  1. moments   find hook beats and cliffhanger beats per episode     (LLM or rules)
  2. assemble  turn (hook, cliffhanger) pairs into 20-60s clips       (code: timing, snapping)
  3. judge     score each clip as a cold viewer would see it          (LLM or rules)
  4. rank      checks, dedupe, diversity, final order                 (code)

Division of labour: the model makes taste calls and only names event IDs; code owns all
timestamps, durations and cut points, so a model can never hand back a time that is not
in the script.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Callable

from . import rules
from .checks import run_checks
from .parse import Episode, Event, fmt_ts

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0


@dataclass
class Moment:
    kind: str           # hook | cliff
    event_id: str
    episode: int
    strength: int       # 1..5
    type: str = ""
    note: str = ""


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    first_event: str = ""
    last_event: str = ""

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self):
        return {"episode": self.episode, "start": fmt_ts(self.start), "end": fmt_ts(self.end),
                "start_sec": round(self.start, 2), "end_sec": round(self.end, 2),
                "duration": self.duration, "first_event": self.first_event,
                "last_event": self.last_event}


@dataclass
class Promo:
    id: str
    kind: str                       # continuous | stitched | cross_episode
    segments: list[Segment]
    hook_event: str
    cliff_event: str
    hook_moment: dict = field(default_factory=dict)
    cliff_moment: dict = field(default_factory=dict)
    prescore: float = 0.0
    judgment: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    score: float = 0.0
    status: str = ""                # recommended | backup | rejected
    rank: int | None = None

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})

    def to_dict(self):
        d = asdict(self)
        d["segments"] = [s.to_dict() for s in self.segments]
        d["duration"] = self.duration
        d["episodes"] = self.episodes
        return d


class Index:
    """Lookup helpers over all parsed episodes."""

    def __init__(self, episodes: list[Episode]):
        self.episodes = {e.number: e for e in episodes}
        self.order = [e.number for e in episodes]
        self.pos = {}
        for e in episodes:
            for i, ev in enumerate(e.events):
                self.pos[ev.id] = (e.number, i)
        self.names = sorted({ev.speaker.split(" (")[0].title() for e in episodes for ev in e.events
                             if ev.type == "dialogue" and len(ev.speaker.split()) <= 2})

    def event(self, eid: str) -> Event:
        ep, i = self.pos[eid]
        return self.episodes[ep].events[i]

    def next_episode(self, n: int) -> int | None:
        i = self.order.index(n)
        return self.order[i + 1] if i + 1 < len(self.order) else None


# ---------------------------------------------------------------------------------- 1. moments
def render_episode(ep: Episode, cam_chars: int = 140) -> str:
    out = [f"=== EPISODE {ep.number} (length {fmt_ts(ep.duration)}) ===", f"Logline: {ep.logline}"]
    for ev in ep.events:
        text = ev.text if ev.type != "camera" else ev.text[:cam_chars]
        if ev.type == "dialogue":
            line = f'{ev.label()}: "{text}"'
        else:
            tag = {"camera": "CAM", "action": "ACTION", "sfx": "SFX"}.get(ev.type, "NOTE")
            line = f"{tag}: {text}"
        out.append(f"[{ev.id} {fmt_ts(ev.start)[:-1]}] {line}")
    return "\n".join(out)


def moments_rules(ep: Episode, k: int = 5) -> list[Moment]:
    hooks, cliffs = [], []
    for i, ev in enumerate(ep.events):
        h = rules.hook_score(ev)
        c = rules.cliff_score(ev, ep, i)
        if h > 0.25:
            hooks.append((h, ev))
        if c > 0.25 and ev.type != "camera":
            cliffs.append((c, ev))
    out = []
    for kind, pool in (("hook", hooks), ("cliff", cliffs)):
        pool.sort(key=lambda x: -x[0])
        chosen = []
        for s, ev in pool:          # keep picks >8s apart so we don't return one beat five times
            if all(abs(ev.start - o.start) > 8 for _, o in chosen):
                chosen.append((s, ev))
            if len(chosen) == k:
                break
        out += [Moment(kind, ev.id, ep.number, rules.to5(s), ev.type, "rules") for s, ev in chosen]
    return out


# (the LLM version of this step lives in runner.py, with validation and on-disk state)


# --------------------------------------------------------------------------------- 2. assemble
def _cam_starts(ep: Episode):
    return [e.start for e in ep.events if e.type == "camera"]


def clip_start(ep: Episode, hook: Event) -> float:
    """Start on the camera cut that carries the hook if it is <=2s earlier, else just before it."""
    starts = [s for s in _cam_starts(ep) if s <= hook.start]
    if starts and hook.start - max(starts) <= 2.0:
        return max(starts)
    return max(0.0, hook.start - 0.3)


def clip_end(ep: Episode, cliff: Event) -> float:
    """Cut on the next camera cut if it lands within 1.5s of the cliff beat, else right after it."""
    ends = [e.end for e in ep.events if e.type == "camera" and cliff.end <= e.end <= cliff.end + 1.5
            and "black" not in e.text.lower()]          # never pad a promo with a black frame
    end = min(ends) if ends else cliff.end + 0.3
    # never let the next line (often the answer) leak into the clip
    nxt = [e.start for e in ep.events if e.type == "dialogue" and e.start >= cliff.end - 0.05
           and e.id != cliff.id]
    if nxt:
        end = min(end, max(cliff.end, min(nxt) - 0.05))
    return min(end, ep.duration)


def _seg(ep: Episode, start: float, end: float) -> Segment:
    inside = [e for e in ep.events if e.start >= start - 0.05 and e.start < end]
    return Segment(ep.number, round(start, 2), round(end, 2),
                   inside[0].id if inside else "", inside[-1].id if inside else "")


def hook_segment(ep: Episode, hook_i: int, min_len=7.0, max_len=18.0) -> Segment:
    hook = ep.events[hook_i]
    start = clip_start(ep, hook)
    ends = [e.end for e in ep.events[hook_i:] if e.type in ("dialogue", "action")]
    end = next((x for x in ends if x - start >= min_len), None)
    if end is None or end - start > max_len:
        end = max(hook.end, start + min_len) if hook.end - start <= max_len else start + max_len
    return _seg(ep, start, min(end, ep.duration))


def cliff_segment(ep: Episode, cliff_i: int, not_before: float, target: float, max_len: float) -> Segment | None:
    cliff = ep.events[cliff_i]
    end = clip_end(ep, cliff)
    # candidate starts: camera cuts or line starts at/before the cliff beat
    starts = sorted({e.start for e in ep.events[:cliff_i + 1]
                     if e.type in ("camera", "dialogue") and e.start >= not_before}, reverse=True)
    best = None
    for s in starts:
        length = end - s
        if length > max_len:
            break
        best = s
        if length >= target:
            break
    if best is None or end - best < 4.0:
        return None
    return _seg(ep, best, end)


def assemble(idx: Index, moments: list[Moment], per_ep_hooks=3, per_ep_cliffs=3,
             cross_hooks=2, cross_cliffs=2) -> list[Promo]:
    by_ep: dict[int, dict[str, list[Moment]]] = {}
    for m in moments:
        by_ep.setdefault(m.episode, {"hook": [], "cliff": []})[m.kind].append(m)
    for d in by_ep.values():
        for k in d:
            d[k].sort(key=lambda m: -m.strength)

    promos, seen = [], set()

    def add(kind, segs, h: Moment, c: Moment):
        key = (h.event_id, c.event_id)
        if key in seen or not segs:
            return
        total = sum(s.duration for s in segs)
        if not (MIN_LEN <= total <= MAX_LEN):
            return
        seen.add(key)
        promos.append(Promo(id="", kind=kind, segments=segs, hook_event=h.event_id,
                            cliff_event=c.event_id, hook_moment=asdict(h), cliff_moment=asdict(c)))

    for ep_no, d in sorted(by_ep.items()):
        ep = idx.episodes[ep_no]
        for h in d["hook"][:per_ep_hooks]:
            _, hi = idx.pos[h.event_id]
            hook = ep.events[hi]
            for c in d["cliff"][:per_ep_cliffs]:
                _, ci = idx.pos[c.event_id]
                cliff = ep.events[ci]
                if cliff.start <= hook.start:
                    continue
                start, end = clip_start(ep, hook), clip_end(ep, cliff)
                span = end - start
                if MIN_LEN <= span <= MAX_LEN:
                    add("continuous", [_seg(ep, start, end)], h, c)
                elif span > MAX_LEN:
                    hs = hook_segment(ep, hi)
                    cs = cliff_segment(ep, ci, not_before=hs.end + 3.0,
                                       target=max(12.0, MIN_LEN - hs.duration),
                                       max_len=MAX_LEN - hs.duration)
                    if cs:
                        add("stitched", [hs, cs], h, c)
                # span < MIN_LEN: too short to stand alone; the stitched/cross paths cover it
        # cross-episode: hook here, cliffhanger in the next episode (series picks up directly)
        nxt = idx.next_episode(ep_no)
        if nxt and nxt in by_ep:
            nep = idx.episodes[nxt]
            for h in d["hook"][:cross_hooks]:
                _, hi = idx.pos[h.event_id]
                hs = hook_segment(ep, hi)
                for c in by_ep[nxt]["cliff"][:cross_cliffs]:
                    _, ci = idx.pos[c.event_id]
                    cs = cliff_segment(nep, ci, not_before=0.0,
                                       target=max(12.0, MIN_LEN - hs.duration),
                                       max_len=MAX_LEN - hs.duration)
                    if cs:
                        add("cross_episode", [hs, cs], h, c)
    return promos


def duration_fit(d: float) -> float:
    if 25 <= d <= 45:
        return 1.0
    if d < 25:
        return max(0.0, (d - MIN_LEN) / 5)
    return max(0.0, (MAX_LEN - d) / 15)


def position_penalty(ep: int, idx: Index, early_bias: float) -> float:
    """Promos from early episodes pay off sooner for a new viewer (and need fewer unlocks)."""
    first, last = idx.order[0], idx.order[-1]
    return early_bias * (ep - first) / max(1, last - first)


def prescore(p: Promo, idx: Index, early_bias: float) -> float:
    kind_bonus = {"continuous": 0.05, "stitched": 0.0, "cross_episode": -0.03}[p.kind]
    return round(0.45 * (p.hook_moment["strength"] - 1) / 4 + 0.40 * (p.cliff_moment["strength"] - 1) / 4
                 + 0.10 * duration_fit(p.duration) + kind_bonus
                 - position_penalty(p.episodes[0], idx, early_bias), 4)


def overlap(a: Promo, b: Promo) -> float:
    """Overlap of the two clips' footage as a share of the shorter clip."""
    inter = 0.0
    for s in a.segments:
        for t in b.segments:
            if s.episode == t.episode:
                inter += max(0.0, min(s.end, t.end) - max(s.start, t.start))
    return inter / max(1e-6, min(a.duration, b.duration))


def dedupe(promos: list[Promo], key: Callable, thresh: float = 0.5) -> list[Promo]:
    kept = []
    for p in sorted(promos, key=key, reverse=True):
        if all(overlap(p, q) < thresh for q in kept):
            kept.append(p)
    return kept


# ------------------------------------------------------------------------------------ 3. judge
def clip_events(p: Promo, idx: Index) -> list[tuple[float, Event, int]]:
    """(time from promo start, event, segment index) for everything that starts inside the clip."""
    out, offset = [], 0.0
    for si, s in enumerate(p.segments):
        ep = idx.episodes[s.episode]
        for ev in ep.events:
            if s.start - 0.05 <= ev.start < s.end:
                out.append((round(offset + ev.start - s.start, 1), ev, si))
        offset += s.duration
    return out


def render_candidate(p: Promo, idx: Index) -> str:
    eps = ", ".join(map(str, p.episodes))
    lines = [f"### {p.id} (episode {eps}, {p.duration:.0f}s, {len(p.segments)} segment(s))"]
    cur = 0
    for t, ev, si in clip_events(p, idx):
        if si != cur:
            lines.append(f"--- cut --- (now episode {p.segments[si].episode})")
            cur = si
        if ev.type == "dialogue":
            lines.append(f'[{t:.1f}s] {ev.label()}: "{ev.text}"')
        else:
            tag = {"camera": "SHOT", "action": "ACTION", "sfx": "SFX"}.get(ev.type, "NOTE")
            lines.append(f"[{t:.1f}s] {tag}: {ev.text}")
    return "\n".join(lines)


def judge_rules(p: Promo, idx: Index) -> dict:
    evs = clip_events(p, idx)
    first = [ev for t, ev, _ in evs if t <= HOOK_WINDOW]
    hook_ev = idx.event(p.hook_event)
    ep_no, ci = idx.pos[p.cliff_event]
    cliff_ev = idx.event(p.cliff_event)
    h = max([rules.hook_score(e) for e in first] + [0.0])
    c = rules.cliff_score(cliff_ev, idx.episodes[ep_no], ci)
    s = rules.standalone_score([ev for _, ev, _ in evs], set(idx.names))
    return {"promo_id": p.id, "hook_score": rules.to5(h), "cliffhanger_score": rules.to5(c),
            "standalone_score": rules.to5(s),
            "hook": _short(hook_ev), "cliffhanger": _short(cliff_ev),
            "why": f"rules: {p.hook_moment.get('type')} opener, cut on {cliff_ev.type}",
            "issues": [], "judge": "rules"}


def _short(ev: Event, n: int = 90) -> str:
    who = f"{ev.speaker.title()}: " if ev.type == "dialogue" else ""
    t = ev.text if len(ev.text) <= n else ev.text[:n].rsplit(" ", 1)[0] + "..."
    return who + t


# ------------------------------------------------------------------------------------- 4. rank
def final_score(p: Promo, idx: Index, early_bias: float) -> float:
    j = p.judgment
    q = (0.35 * (j["hook_score"] - 1) + 0.35 * (j["cliffhanger_score"] - 1)
         + 0.30 * (j["standalone_score"] - 1)) / 4
    return round(q + 0.05 * duration_fit(p.duration) - position_penalty(p.episodes[0], idx, early_bias), 4)


def rank(promos: list[Promo], idx: Index, early_bias: float, per_episode_cap: int = 2,
         min_scores: int = 3) -> list[Promo]:
    for p in promos:
        p.checks = run_checks(p, idx)
        p.score = final_score(p, idx, early_bias)
        hard_ok = all(c["ok"] for c in p.checks.values() if c["level"] == "hard")
        soft_ok = all(p.judgment[k] >= min_scores for k in ("hook_score", "cliffhanger_score", "standalone_score"))
        p.status = "recommended" if hard_ok and soft_ok else ("backup" if hard_ok else "rejected")
    order = {"recommended": 0, "backup": 1, "rejected": 2}
    live = [p for p in promos if p.status != "rejected"]
    kept = dedupe(live, key=lambda p: (-order[p.status], p.score))
    per_ep: dict[int, int] = {}
    used_hooks: set[str] = set()      # same opening beat = same promo to a scrolling viewer
    ranked = []
    for p in sorted(kept, key=lambda p: (order[p.status], -p.score)):
        e = p.episodes[0]
        if per_ep.get(e, 0) >= per_episode_cap or p.hook_event in used_hooks:
            continue
        per_ep[e] = per_ep.get(e, 0) + 1
        used_hooks.add(p.hook_event)
        ranked.append(p)
    for i, p in enumerate(ranked, 1):
        p.rank = i
    return ranked


# --------------------------------------------------------------------------------------- run
@dataclass
class RunResult:
    ranked: list[Promo]
    candidates: list[Promo]
    moments: list[Moment]
    stats: dict


def build_candidates(idx: Index, moments: list[Moment], early_bias: float) -> list[Promo]:
    """Moments -> every valid 20-60s clip, with stable IDs (P001...) and a pre-score."""
    cands = assemble(idx, moments)
    for i, p in enumerate(sorted(cands, key=lambda p: (p.episodes[0], p.segments[0].start)), 1):
        p.id = f"P{i:03d}"
    for p in cands:
        p.prescore = prescore(p, idx, early_bias)
    return cands


def judge_pool(cands: list[Promo], top_k: int | None = None) -> list[Promo]:
    """Clips worth judging: drop near-duplicates, best pre-score first, optionally top-k."""
    pool = sorted(dedupe(cands, key=lambda p: p.prescore, thresh=0.7), key=lambda p: -p.prescore)
    return pool[:top_k] if top_k else pool


def finish(cands: list[Promo], judged: list[Promo], idx: Index, early_bias: float) -> list[Promo]:
    """Rank the judged clips. Anything not judged is labelled, never scored."""
    judged_ids = {p.id for p in judged}
    for p in cands:
        if p.id not in judged_ids:
            p.status = "not_judged"   # overlapped a better clip, below judge_top_k, or no model result
    return rank(judged, idx, early_bias)


def run(episodes: list[Episode], series: str = "", premise: str = "", early_bias: float = 0.10,
        log: Callable = print) -> RunResult:
    """No-API baseline: keyword rules find the moments and judge the clips.
    (The LLM version is runner.run_llm.)"""
    idx = Index(episodes)
    stats: dict = {"episodes": len(episodes), "mode": "rules", "complete": True}

    log("1/4 finding hook and cliffhanger moments (rules)")
    moments = [m for e in episodes for m in moments_rules(e)]
    stats["n_moments"] = len(moments)

    log("2/4 assembling 20-60s candidate clips")
    cands = build_candidates(idx, moments, early_bias)
    stats["n_candidates"] = len(cands)

    log("3/4 judging candidates (rules)")
    pool = judge_pool(cands)
    for p in pool:
        p.judgment = judge_rules(p, idx)
    stats["n_judged"] = len(pool)

    log("4/4 checks and ranking")
    ranked = finish(cands, pool, idx, early_bias)
    stats["n_ranked"] = len(ranked)
    stats["n_recommended"] = sum(p.status == "recommended" for p in ranked)
    return RunResult(ranked, cands, moments, stats)


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
