"""Parse Kuku-style microdrama screenplays into timestamped events.

Expected format (one .txt per episode):

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. CATHEDRAL – DAY
    Scene start: 00:00.00
    Scene end: 02:18.00
    **CAMERA:**
    Start: (00:00.00) | End: (00:04.80)
    text...
    **LADY JULIANA (Confused, Breathless):**
    Start: (00:13.50) | End: (00:14.30)
    "Why?"
    --- SCRIPT END ---

Timestamps are episode-relative (mm:ss.ss). The parser is tolerant: unknown
blocks are kept as type "other", and blocks without a time line are dropped
and counted in `Episode.warnings`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?")
BLOCK_LABEL_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
START_END_RE = re.compile(
    r"^Start:\s*\(?\s*([\d:.]+)\s*\)?\s*\|\s*End:\s*\(?\s*([\d:.]+)\s*\)?\s*$"
)
EP_NUM_RE = re.compile(r"(\d+)")
NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx", "MUSIC": "sfx"}
VOICE_TAGS = {"O.S.", "V.O.", "O.C.", "CONT'D", "CONT’D"}


def to_seconds(ts: str) -> float:
    m = TIME_RE.search(ts)
    if not m:
        raise ValueError(f"bad timestamp: {ts!r}")
    minutes, seconds, frac = m.group(1), m.group(2), m.group(3) or "0"
    return int(minutes) * 60 + int(seconds) + float("0." + frac)


def fmt_ts(sec: float) -> str:
    sec = max(0.0, sec)
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"


@dataclass
class Event:
    id: str            # stable id, e.g. "E01-042"; LLMs refer to events by id, never by time
    episode: int
    scene: int
    type: str          # camera | action | sfx | dialogue | other
    start: float
    end: float
    text: str
    speaker: str = ""
    tags: str = ""     # delivery / emotion tags from the label, e.g. "Outraged, Accusing"

    def label(self) -> str:
        if self.type == "dialogue":
            return f"{self.speaker} ({self.tags})" if self.tags else self.speaker
        return self.type.upper()


@dataclass
class Scene:
    number: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    number: int
    title: str
    logline: str
    continuity: str
    source: str
    scenes: list[Scene] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [s.end for s in self.scenes] + [e.end for e in self.events]
        return max(ends) if ends else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


def _split_label(label: str) -> tuple[str, str, str]:
    """Return (type, speaker, tags) for a block label like 'MATRIARCH (O.S.) (Firm)'."""
    up = label.strip().upper()
    if up in NON_DIALOGUE:
        return NON_DIALOGUE[up], "", ""
    parens = re.findall(r"\(([^)]*)\)", label)
    speaker = re.sub(r"\([^)]*\)", "", label).strip()
    tags = [p.strip() for p in parens if p.strip().upper() not in VOICE_TAGS]
    voice = [p.strip() for p in parens if p.strip().upper() in VOICE_TAGS]
    if voice:
        speaker = f"{speaker} ({', '.join(voice)})"
    return "dialogue", speaker, ", ".join(tags)


def parse_text(text: str, source: str = "", fallback_number: int = 0) -> Episode:
    lines = text.replace("\r\n", "\n").split("\n")
    header = {}
    for ln in lines[:20]:
        for key in ("EPISODE NUMBER", "EPISODE LOGLINE", "CONTINUITY NOTES"):
            if ln.startswith(key + ":"):
                header[key] = ln.split(":", 1)[1].strip()
    title = header.get("EPISODE NUMBER", "")
    m = EP_NUM_RE.search(title)
    number = int(m.group(1)) if m else fallback_number
    ep = Episode(number=number, title=title, logline=header.get("EPISODE LOGLINE", ""),
                 continuity=header.get("CONTINUITY NOTES", ""), source=source)

    scene_no, heading, s_start, s_end = 0, "", None, None
    i, n, counter = 0, len(lines), 0

    def close_scene():
        if scene_no and s_start is not None:
            ep.scenes.append(Scene(scene_no, heading, s_start, s_end if s_end is not None else s_start))

    while i < n:
        ln = lines[i].strip()
        if re.match(r"^SCENE\s+\d+", ln):
            close_scene()
            scene_no = int(re.findall(r"\d+", ln)[0])
            heading, s_start, s_end = "", None, None
        elif ln.startswith("Scene heading:"):
            heading = ln.split(":", 1)[1].strip()
        elif ln.startswith("Scene start:"):
            s_start = to_seconds(ln.split(":", 1)[1])
        elif ln.startswith("Scene end:"):
            s_end = to_seconds(ln.split(":", 1)[1])
        else:
            bm = BLOCK_LABEL_RE.match(ln)
            if bm:
                label = bm.group(1)
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                tm = START_END_RE.match(lines[j].strip()) if j < n else None
                if not tm:
                    ep.warnings.append(f"block without time at line {i + 1}: {label}")
                    i += 1
                    continue
                start, end = to_seconds(tm.group(1)), to_seconds(tm.group(2))
                body = []
                k = j + 1
                while k < n and lines[k].strip() and not BLOCK_LABEL_RE.match(lines[k].strip()) \
                        and not lines[k].startswith("--- "):
                    body.append(lines[k].strip())
                    k += 1
                btype, speaker, tags = _split_label(label)
                content = " ".join(body).strip()
                if btype == "dialogue":
                    content = content.strip('"“” ')
                if end < start:
                    ep.warnings.append(f"end<start at line {i + 1}; swapped")
                    start, end = end, start
                counter += 1
                ep.events.append(Event(id=f"E{number:02d}-{counter:03d}", episode=number,
                                       scene=scene_no, type=btype, start=start, end=end,
                                       text=content, speaker=speaker, tags=tags))
                i = k
                continue
        i += 1
    close_scene()
    if not ep.events:
        ep.warnings.append("no timed blocks found")
    return ep


def load_episodes(folder: str | Path) -> list[Episode]:
    """Load every *.txt screenplay under `folder` (recursively), sorted by episode number."""
    paths = sorted(Path(folder).rglob("*.txt"))
    episodes = []
    for idx, p in enumerate(paths, 1):
        ep = parse_text(p.read_text(encoding="utf-8", errors="replace"), source=p.name, fallback_number=idx)
        if ep.events:
            episodes.append(ep)
    episodes.sort(key=lambda e: e.number)
    return episodes
