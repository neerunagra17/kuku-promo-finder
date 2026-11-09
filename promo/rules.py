"""Rule-based scorers. Used as (a) the no-LLM baseline and (b) a cheap pre-filter.

Scores are 0..1. They look only at the text of one event plus its neighbours,
so they are fast, deterministic and explainable - and blind to story context,
which is exactly where they fail (see docs/solution doc, "where it picks badly").
"""
from __future__ import annotations

import re

from .parse import Episode, Event

# Physical / visual shocks that read in 1-3 seconds with no context.
SHOCK_ACTION = {
    "slap": 1.0, "slaps": 1.0, "strikes": 0.9, "struck": 0.9, "punch": 0.9, "hits": 0.6,
    "crop": 0.7, "whip": 0.8, "grabs": 0.6, "seize": 0.7, "seizes": 0.7, "drag": 0.6,
    "drags": 0.6, "shoves": 0.7, "throws": 0.6, "hurls": 0.7, "explodes": 1.0,
    "explosion": 1.0, "fireball": 1.0, "blood": 0.8, "bleeding": 0.8, "knife": 0.8,
    "blade": 0.7, "sword": 0.7, "pistol": 0.8, "gun": 0.8, "flames": 0.8, "ablaze": 0.9,
    "fire": 0.5, "burns": 0.7, "branding": 0.9, "chains": 0.6, "chained": 0.6,
    "kneel": 0.6, "kneels": 0.6, "one knee": 0.8, "ring": 0.3, "kiss": 0.7, "kisses": 0.7,
    "collapses": 0.7, "faints": 0.7, "screams": 0.7, "shrieking": 0.6, "smashes": 0.8,
    "shatters": 0.7, "arrest": 0.7, "noose": 0.9, "gallows": 0.8, "mud": 0.4,
    "tears": 0.3, "gasps": 0.4, "storms": 0.6, "bursts": 0.6, "kicks": 0.8,
}
# Words that make a line land as a reveal / threat / insult on its own.
SHOCK_WORDS = {
    "marrying": 0.7, "married": 0.5, "wife": 0.4, "husband": 0.4, "duke": 0.6,
    "duchess": 0.4, "dead": 0.6, "alive": 0.6, "kill": 0.8, "die": 0.7, "hang": 0.8,
    "execution": 0.8, "witch": 0.8, "witchcraft": 0.8, "poison": 0.8, "poisoned": 0.8,
    "arsenic": 0.7, "betray": 0.7, "traitor": 0.8, "liar": 0.6, "thief": 0.7,
    "worthless": 0.8, "nothing": 0.4, "plain": 0.4, "disowned": 0.8, "murder": 0.9,
    "pregnant": 0.9, "son": 0.3, "mother": 0.3, "lie": 0.4, "guards": 0.5, "arrest": 0.7,
    "seize": 0.6, "invalid": 0.6, "never": 0.3, "beg": 0.6, "kneel": 0.6, "burn": 0.7,
}
INTENSE_TAGS = {
    "screaming": 1.0, "shouting": 0.9, "furious": 0.9, "outraged": 0.8, "venomous": 0.8,
    "gloating": 0.7, "cruel": 0.8, "sadistic": 0.9, "menacing": 0.8, "threatening": 0.9,
    "mocking": 0.6, "spiteful": 0.6, "scornful": 0.6, "desperate": 0.7, "sobbing": 0.6,
    "crying": 0.6, "shocked": 0.7, "horrified": 0.8, "roaring": 0.9, "seething": 0.7,
    "commanding": 0.5, "cold": 0.4, "icy": 0.5, "dismissive": 0.4, "contemptuous": 0.7,
    "taunting": 0.6, "accusing": 0.7, "defiant": 0.5, "fierce": 0.6, "smug": 0.5,
}
CALM_TAGS = {"gentle", "soft", "tender", "warm", "reflective", "calm", "soothing", "relieved",
             "quiet", "affectionate", "reassuring", "continuing"}
OPEN_WORDS = ("will you", "who are you", "what do you", "or else", "or watch", "unless",
              "if you", "before", "tomorrow", "you will", "i will", "we will", "until",
              "who is", "where is", "what have", "how dare", "who did", "what is")
ARRIVAL = ("door swings open", "door opens", "bursts in", "storms in", "enters", "arrives",
           "footsteps", "silhouette", "figure appears", "latch", "knock")
CLIFF_SFX = ("cliffhanger", "sting", "ominous", "swells", "abrupt", "freeze", "cut to black")

_word = re.compile(r"[a-z']+")


def _lex(text: str, table: dict) -> float:
    t = text.lower()
    hits = [w for w in table if (" " in w and w in t) or w in set(_word.findall(t))]
    if not hits:
        return 0.0
    top = sorted((table[w] for w in hits), reverse=True)
    return min(1.0, top[0] + 0.15 * sum(top[1:3]))


def _tag_intensity(tags: str) -> float:
    words = set(_word.findall(tags.lower()))
    inten = max([INTENSE_TAGS.get(w, 0.0) for w in words] + [0.0])
    if words & CALM_TAGS:
        inten -= 0.3
    return max(0.0, inten)


def hook_score(ev: Event) -> float:
    """How likely this event, played in the first 3s, stops a scroller."""
    t = ev.text
    if ev.type in ("action", "camera"):
        s = _lex(t, SHOCK_ACTION) * (1.0 if ev.type == "action" else 0.8)
    elif ev.type == "dialogue":
        words = len(t.split())
        s = 0.45 * _tag_intensity(ev.tags) + 0.45 * _lex(t, SHOCK_WORDS)
        if "!" in t:
            s += 0.12
        if words <= 9:
            s += 0.1          # short lines land inside 3 seconds
        if words > 22:
            s -= 0.2
        if re.match(r"^(you|how dare|get out|liar|enough|stop|no)\b", t.lower()):
            s += 0.12
    elif ev.type == "sfx":
        s = 0.5 * _lex(t, {"slap": 1, "crack": 1, "boom": 1, "explosion": 1, "scream": 1, "gasp": 0.6})
    else:
        s = 0.0
    return max(0.0, min(1.0, s))


def cliff_score(ev: Event, ep: Episode, idx: int) -> float:
    """How good a place this is to CUT, leaving the viewer needing the next moment."""
    t = ev.text.lower()
    s = 0.0
    if ev.type == "dialogue":
        s += 0.35 * _tag_intensity(ev.tags) + 0.25 * _lex(t, SHOCK_WORDS)
        if t.rstrip().endswith("?"):
            s += 0.3
        if any(w in t for w in OPEN_WORDS):
            s += 0.25
        if t.rstrip().endswith(("...", "—", "-")):
            s += 0.2
    elif ev.type in ("action", "camera"):
        s += 0.5 * _lex(t, SHOCK_ACTION)
        if any(a in t for a in ARRIVAL):
            s += 0.35
    elif ev.type == "sfx":
        s += 0.3 + (0.3 if any(c in t for c in CLIFF_SFX) else 0.0)
    # Screenwriters put the strongest unresolved beat at the episode end.
    remaining = ep.duration - ev.end
    if remaining <= 6:
        s += 0.45
    elif remaining <= 15:
        s += 0.15
    # A cut point is weak if the very next line answers it.
    nxt = next((e for e in ep.events[idx + 1:] if e.type == "dialogue"), None)
    if nxt and ev.type == "dialogue" and nxt.start - ev.end < 1.0 and nxt.speaker != ev.speaker:
        s -= 0.1
    return max(0.0, min(1.0, s))


STANDALONE_ANCHORS = ("husband", "wife", "sister", "mother", "father", "fiancé", "fiance",
                      "married", "wedding", "duke", "duchess", "maid", "family", "son",
                      "daughter", "bride", "groom", "lady", "lord", "my ")


def standalone_score(events: list[Event], known_names: set[str]) -> float:
    """Crude proxy: lines that state relationships help; many named people hurt."""
    text = " ".join(e.text for e in events).lower()
    anchors = sum(1 for a in STANDALONE_ANCHORS if a in text)
    speakers = {e.speaker for e in events if e.type == "dialogue"}
    names = {n for n in known_names if n.lower() in text}
    s = 0.5 + 0.08 * min(anchors, 4) - 0.07 * max(0, len(speakers) - 3) - 0.05 * max(0, len(names) - 2)
    return max(0.0, min(1.0, s))


def to5(x: float) -> int:
    """Map 0..1 to a 1..5 score."""
    return int(max(1, min(5, round(1 + 4 * x))))
