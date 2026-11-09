"""LLM run: test request first, bounded concurrency, resumable on-disk state, no made-up results.

  test request -> moments (N episodes per call) -> assemble clips (code) -> judge -> rank (code)

- Before any episode is sent, one tiny request checks the key, endpoint and model. If the model
  is down it tries --fallback-models in order; the model that answers is pinned for this run.
- A result is saved only from a valid model reply. A failed request saves nothing; its episodes
  or clips stay pending. Nothing is filled in by rules, defaults or guesses.
- Every saved result is its own file under <out>/state/, so a rerun skips finished work.
- If the API still fails after all retries, the run stops instead of hammering an overloaded
  server, and reports what is pending. Rerun the same command later to continue.
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .llm import BaseLLM, FatalError, LLMError, ServiceUnavailable, Stopped, UnitError, normalize_model
from .parse import Episode
from .pipeline import (Index, Moment, Promo, RunResult, build_candidates, finish, judge_pool,
                       render_candidate, render_episode)
from .prompts import JUDGE_SYSTEM, JUDGE_USER, MOMENTS_SYSTEM, MOMENTS_USER
from .state import StateStore, pack_fingerprint

PROMPT_VERSION = hashlib.sha1("".join([MOMENTS_SYSTEM, MOMENTS_USER, JUDGE_SYSTEM, JUDGE_USER])
                              .encode()).hexdigest()[:10]
OUTPUT_FILES = ("promos_ranked.json", "promos_ranked.csv", "all_candidates.json", "moments.json",
                "run_stats.json", "editor_review_sheet.csv", "report.md", "run_status.json")


@dataclass
class LLMRun:
    complete: bool
    result: RunResult | None          # None unless complete (or allow_partial)
    stats: dict
    pending_episodes: list = field(default_factory=list)
    pending_clips: list = field(default_factory=list)
    stop_reason: str = ""


class _Counter:
    def __init__(self):
        self._d: dict = {}
        self._lock = threading.Lock()

    def add(self, k: str, n: int = 1):
        with self._lock:
            self._d[k] = self._d.get(k, 0) + n

    def as_dict(self) -> dict:
        with self._lock:
            return dict(self._d)


# ------------------------------------------------------------------------------ validation
def _as_int(x):
    try:
        f = float(str(x).strip().lower().replace("episode", "").strip())
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else None


def _score(x):
    v = _as_int(x)
    return v if v is not None and 1 <= v <= 5 else None


def _validate_moments(item: dict, n: int, idx: Index) -> tuple[list[Moment], dict]:
    out = []
    d = {"returned": 0, "invalid_ids": 0, "wrong_episode_ids": 0, "invalid_fields": 0}
    for kind, key in (("hook", "hooks"), ("cliff", "cliffhangers")):
        items = item.get(key) or []
        if not isinstance(items, list):
            d["invalid_fields"] += 1
            continue
        for m in items[:6]:
            d["returned"] += 1
            if not isinstance(m, dict):
                d["invalid_fields"] += 1
                continue
            eid = str(m.get("id", "")).strip()
            if eid not in idx.pos:
                d["invalid_ids"] += 1            # invented or malformed ID: dropped, never guessed
                continue
            if idx.pos[eid][0] != n:
                d["wrong_episode_ids"] += 1
                continue
            strength = _score(m.get("strength"))
            if strength is None:
                d["invalid_fields"] += 1
                continue
            out.append(Moment(kind, eid, n, strength, str(m.get("type", ""))[:40],
                              str(m.get("why") or m.get("open_question") or "")[:200]))
    return out, d


def _validate_judgment(j: dict) -> dict | None:
    """Return a clean judgment, or None if anything required is missing or out of range."""
    scores = {k: _score(j.get(k)) for k in ("hook_score", "cliffhanger_score", "standalone_score")}
    if None in scores.values():
        return None
    hook, cliff = str(j.get("hook") or "").strip(), str(j.get("cliffhanger") or "").strip()
    if not hook or not cliff:
        return None
    issues = j.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    return {**scores, "hook": hook[:240], "cliffhanger": cliff[:240],
            "why": str(j.get("why") or "").strip()[:240], "issues": [str(x)[:160] for x in issues][:5]}


def clip_key(p: Promo) -> str:
    raw = json.dumps([[(s.episode, s.start, s.end) for s in p.segments], p.hook_event, p.cliff_event,
                      PROMPT_VERSION])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _span(nums) -> str:
    nums = sorted(nums)
    if not nums:
        return "-"
    return f"{nums[0]}-{nums[-1]}" if len(nums) > 1 and nums == list(range(nums[0], nums[-1] + 1)) \
        else ", ".join(map(str, nums))


# ------------------------------------------------------------------------------- preflight
def preflight(llm: BaseLLM, fallbacks=(), log: Callable = print, retries: int = 4) -> str:
    """One tiny request before any episode is sent. Tries llm.model, then each fallback
    that the key can see; returns the first model that answers and leaves llm set to it."""
    try:
        available = llm.list_models()
    except FatalError:
        raise
    except LLMError as e:
        log(f"  could not list models ({e}); skipping the name check")
        available = None
    if available is not None:
        log(f"  key accepted; {len(available)} models available")
    chain: list[str] = []
    for m in [llm.model, *fallbacks]:
        m = normalize_model(m)
        if m and m not in chain:
            chain.append(m)
    tried = []
    for m in chain:
        if available is not None and m not in available:
            log(f"  {m}: not available for this key - skipped")
            tried.append(f"{m}: not in the model list for this key")
            continue
        llm.model = m
        log(f"  {m}: test request ...")
        try:
            rep = llm.generate_json("Reply with JSON only.", 'Return exactly this JSON object: {"ok": true}',
                                    {"task": "preflight"}, max_retries=retries)
        except FatalError as e:
            if e.kind == "auth":
                raise
            tried.append(str(e))
            log(f"  failed: {e}")
            continue
        except (ServiceUnavailable, UnitError) as e:
            tried.append(str(e) if str(e).startswith(m) else f"{m}: {e}")
            log(f"  failed: {tried[-1]}")
            continue
        if rep.data.get("ok") is not True:
            tried.append(f"{m}: unexpected reply {str(rep.data)[:80]}")
            continue
        log(f"  {m}: OK ({rep.seconds:.1f}s)")
        return m
    if all("not in the model list" in t for t in tried):
        raise FatalError("none of these models is available for this key: " + ", ".join(chain) +
                         ". Run `python run.py --check --provider ...` to see the list.", kind="model")
    raise ServiceUnavailable("no model passed the test request:\n    " + "\n    ".join(tried))


# ---------------------------------------------------------------------------- worker pool
def _run_pool(units: list, fn: Callable, llm: BaseLLM, concurrency: int):
    """fn(unit) for every unit with at most `concurrency` requests in flight.
    Returns (unit_failures, stop_error). The first ServiceUnavailable/FatalError stops new
    requests; requests already in flight finish and keep their results."""
    failures, stop_err = [], None
    llm.stop_event.clear()
    ex = ThreadPoolExecutor(max_workers=max(1, int(concurrency)))
    futures = {ex.submit(fn, u): u for u in units}
    try:
        for fut in as_completed(futures):
            try:
                fut.result()
            except Stopped:
                pass
            except UnitError as e:
                failures.append((futures[fut], e))
            except (ServiceUnavailable, FatalError) as e:
                if stop_err is None:
                    stop_err = e
                llm.stop_event.set()
    except BaseException:
        llm.stop_event.set()
        raise
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        llm.stop_event.clear()
    return failures, stop_err


def _check_manifest(man: dict, llm: BaseLLM, fallbacks, fp: str, out_dir):
    problems = []
    if man.get("pack") != fp:
        problems.append("the screenplay pack is different")
    if man.get("prompt_version") != PROMPT_VERSION:
        problems.append("the prompts have changed")
    if man.get("provider") != llm.name:
        problems.append(f"it used provider '{man.get('provider')}'")
    allowed = {normalize_model(m) for m in [llm.model, *fallbacks]}
    if man.get("model") not in allowed:
        problems.append(f"it is pinned to model '{man.get('model')}'")
    if problems:
        raise FatalError(f"{out_dir} already holds a run that does not match this command: "
                         f"{'; '.join(problems)}. Add --fresh to start over in this folder, "
                         f"or use a different --out.", kind="state")


# ------------------------------------------------------------------------------------- run
def run_llm(episodes: list[Episode], llm: BaseLLM, out_dir, *, series: str = "", premise: str = "",
            fallback_models=(), fresh: bool = False, allow_partial: bool = False, concurrency: int = 1,
            episodes_per_call: int = 5, judge_batch: int = 12, judge_top_k: int = 60,
            early_bias: float = 0.10, test_request: bool = True, log: Callable = print) -> LLMRun:
    out_dir = Path(out_dir)
    store = StateStore(out_dir / "state")
    if fresh:
        store.clear()
        for name in OUTPUT_FILES:
            (out_dir / name).unlink(missing_ok=True)
    idx = Index(episodes)
    premise = premise or " ".join(e.logline for e in episodes[:2])
    fallback_models = [normalize_model(m) for m in fallback_models if m]
    fp = pack_fingerprint(episodes)
    manifest = store.manifest()
    if manifest:
        _check_manifest(manifest, llm, fallback_models, fp, out_dir)
        llm.model = manifest["model"]
        done = sum(store.has_moments(e.number) for e in episodes)
        log(f"resuming {out_dir}: model {llm.model} (pinned at the start of this run), "
            f"{done}/{len(episodes)} episodes already done")
    llm.on_call = lambda **row: store.log_usage(**row)
    counts = _Counter()
    api_ready = False

    def ensure_api():
        nonlocal manifest, api_ready
        if api_ready:
            return
        if test_request:
            log("API check: one small test request before any episode is sent")
            preflight(llm, [] if manifest else fallback_models, log)
        if manifest is None:
            manifest = {"provider": llm.name, "model": llm.model,
                        "base_url": getattr(llm, "base_url", ""), "prompt_version": PROMPT_VERSION,
                        "pack": fp, "series": series, "episodes": len(episodes)}
            store.write_manifest(manifest)
        api_ready = True

    def log_failures(task, fails):
        for unit, e in fails:
            ids = [x.number for x in unit] if task == "moments" else [x.id for x in unit]
            store.log_failure(task=task, unit=ids, model=llm.model, error=str(e)[:300])
            log(f"  {task} {_span(ids) if task == 'moments' else ', '.join(ids)}: {e}")

    # ---- 1. moments, per episode ------------------------------------------------------
    def moments_unit(group: list[Episode]):
        nums = [e.number for e in group]
        prompt = MOMENTS_USER.format(n=len(group), series=series,
                                     transcripts="\n\n".join(render_episode(e) for e in group))
        rep = llm.generate_json(MOMENTS_SYSTEM, prompt, {"task": "moments", "episodes": nums})
        items = rep.data.get("episodes")
        if not isinstance(items, list):
            raise UnitError("reply has no 'episodes' list")
        want, saved = set(nums), set()
        for item in items:
            n = _as_int(item.get("episode")) if isinstance(item, dict) else None
            if n not in want or n in saved:
                counts.add("unexpected_episode_entries")
                continue
            moments, dropped = _validate_moments(item, n, idx)
            store.save_moments(n, {"episode": n, "model": llm.model, "prompt_version": PROMPT_VERSION,
                                   "moments": [asdict(m) for m in moments], "dropped": dropped})
            saved.add(n)
        if saved:
            log(f"  moments: episode {_span(saved)} saved ({rep.seconds:.0f}s)")
        missing = sorted(want - saved)
        if missing:
            raise UnitError(f"reply had no entry for episode(s) {missing}")

    stop_err = None
    pending = [e for e in episodes if not store.has_moments(e.number)]
    if pending:
        ensure_api()
        log(f"moments: {len(pending)} of {len(episodes)} episodes to process, "
            f"{episodes_per_call} per request, {concurrency} at a time")
        units = [pending[i:i + episodes_per_call] for i in range(0, len(pending), episodes_per_call)]
        fails, stop_err = _run_pool(units, moments_unit, llm, concurrency)
        log_failures("moments", fails)
        retry = [e for e in episodes if not store.has_moments(e.number)]
        if retry and not stop_err:
            log(f"moments: retrying {len(retry)} episode(s) one per request")
            fails, stop_err = _run_pool([[e] for e in retry], moments_unit, llm, concurrency)
            log_failures("moments", fails)
    pending_eps = [e.number for e in episodes if not store.has_moments(e.number)]

    # ---- 2. assemble from saved moments only -----------------------------------------
    moments: list[Moment] = []
    for e in episodes:
        if store.has_moments(e.number):
            moments += [Moment(**m) for m in store.load_moments(e.number)["moments"]]
    cands = build_candidates(idx, moments, early_bias)
    to_judge = judge_pool(cands, judge_top_k)
    keys = {p.id: clip_key(p) for p in to_judge}

    # ---- 3. judge, per clip -----------------------------------------------------------
    def judge_unit(batch: list[Promo]):
        ids = [p.id for p in batch]
        prompt = JUDGE_USER.format(series=series, premise=premise,
                                   candidates="\n\n".join(render_candidate(p, idx) for p in batch))
        rep = llm.generate_json(JUDGE_SYSTEM, prompt, {"task": "judge", "promo_ids": ids})
        items = rep.data.get("judgments")
        if not isinstance(items, list):
            raise UnitError("reply has no 'judgments' list")
        by_id, saved = {p.id: p for p in batch}, []
        for j in items:
            pid = str(j.get("promo_id", "")).strip() if isinstance(j, dict) else ""
            if pid not in by_id or pid in saved:
                counts.add("unknown_clip_ids")
                continue
            clean = _validate_judgment(j)
            if clean is None:
                counts.add("invalid_judgments")
                continue
            store.save_judgment(keys[pid], {"key": keys[pid], "promo_id_at_save": pid,
                                            "model": llm.model, "judgment": clean})
            saved.append(pid)
        if saved:
            log(f"  judge: {len(saved)} clip(s) saved ({rep.seconds:.0f}s)")
        missing = [i for i in ids if i not in saved]
        if missing:
            raise UnitError(f"no valid judgment for {', '.join(missing)}")

    if not stop_err and (allow_partial or not pending_eps):
        todo = [p for p in to_judge if not store.has_judgment(keys[p.id])]
        if todo:
            ensure_api()
            log(f"judge: {len(todo)} of {len(to_judge)} clips to score, {judge_batch} per request")
            batches = [todo[i:i + judge_batch] for i in range(0, len(todo), judge_batch)]
            fails, stop_err = _run_pool(batches, judge_unit, llm, concurrency)
            log_failures("judge", fails)
            retry = [p for p in to_judge if not store.has_judgment(keys[p.id])]
            if retry and not stop_err:
                size = max(1, judge_batch // 4)
                log(f"judge: retrying {len(retry)} clip(s), {size} per request")
                fails, stop_err = _run_pool([retry[i:i + size] for i in range(0, len(retry), size)],
                                            judge_unit, llm, concurrency)
                log_failures("judge", fails)

    judged = []
    for p in to_judge:
        if store.has_judgment(keys[p.id]):
            rec = store.load_judgment(keys[p.id])
            p.judgment = {**rec["judgment"], "promo_id": p.id, "judge": rec["model"]}
            judged.append(p)
    # clips are only "pending" once judging could run (all episodes done, or partial allowed)
    pending_clips = [p.id for p in to_judge if not p.judgment] if (allow_partial or not pending_eps) else []
    if stop_err:
        store.log_failure(task="run", unit=[], model=llm.model, error=f"stopped: {stop_err}"[:300])
    complete = not pending_eps and not pending_clips
    reason = str(stop_err) if stop_err else ("" if complete else "some model replies were unusable")

    # ---- 4. stats + rank ---------------------------------------------------------------
    mstats = {"returned": 0, "invalid_ids": 0, "wrong_episode_ids": 0, "invalid_fields": 0}
    for e in episodes:
        if store.has_moments(e.number):
            for k, v in store.load_moments(e.number).get("dropped", {}).items():
                mstats[k] = mstats.get(k, 0) + v
    stats = {"episodes": len(episodes), "mode": "llm", "complete": complete,
             "llm": {"provider": llm.name, "model": llm.model, **store.usage_totals(),
                     "this_run": asdict(llm.usage)},
             "moments": {**mstats, "episodes_done": len(episodes) - len(pending_eps),
                         "episodes_pending": pending_eps},
             "judge": {"clips_to_judge": len(to_judge), "judged": len(judged), "pending": pending_clips},
             "validation_drops_this_run": counts.as_dict(),
             "failed_requests_logged": store.failure_count(),
             "stop_reason": reason,
             "n_moments": len(moments), "n_candidates": len(cands), "n_judged": len(judged)}

    if not complete and not allow_partial:
        return LLMRun(False, None, stats, pending_eps, pending_clips, reason)
    ranked = finish(cands, judged, idx, early_bias)
    stats["n_ranked"] = len(ranked)
    stats["n_recommended"] = sum(p.status == "recommended" for p in ranked)
    return LLMRun(complete, RunResult(ranked, cands, moments, stats), stats, pending_eps, pending_clips,
                  reason)
