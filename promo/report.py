"""Write raw outputs: JSON, CSV, a markdown report, and an editor review sheet."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .parse import fmt_ts
from .pipeline import Index, Promo, RunResult, clip_events


def _segments_str(p: Promo) -> str:
    return " + ".join(f"Ep{s.episode} {fmt_ts(s.start)}-{fmt_ts(s.end)}" for s in p.segments)


def _failed(p: Promo) -> str:
    return "; ".join(f"{k} ({v['detail']})" for k, v in p.checks.items() if not v["ok"])


def write_outputs(res: RunResult, idx: Index, out_dir: str | Path, title: str = "") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}

    def dump(name, obj):
        (out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        files[name] = str(out / name)

    dump("promos_ranked.json", [p.to_dict() for p in res.ranked])
    dump("all_candidates.json", [p.to_dict() for p in sorted(res.candidates, key=lambda p: p.id)])
    dump("moments.json", [m.__dict__ for m in res.moments])
    dump("run_stats.json", res.stats)

    cols = ["rank", "status", "promo_id", "kind", "episodes", "segments", "duration_s", "hook",
            "cliffhanger", "why", "hook_score", "cliffhanger_score", "standalone_score", "score",
            "failed_checks", "judge_issues"]
    with open(out / "promos_ranked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in res.ranked:
            j = p.judgment
            w.writerow([p.rank, p.status, p.id, p.kind, " ".join(map(str, p.episodes)),
                        _segments_str(p), p.duration, j.get("hook"), j.get("cliffhanger"), j.get("why"),
                        j.get("hook_score"), j.get("cliffhanger_score"), j.get("standalone_score"),
                        p.score, _failed(p), " | ".join(j.get("issues", []))])
    files["promos_ranked.csv"] = str(out / "promos_ranked.csv")

    with open(out / "editor_review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "promo_id", "segments", "hook", "cliffhanger",
                    "verdict (post / fix / reject)", "fixed start/end if any", "notes"])
        for p in res.ranked:
            w.writerow([p.rank, p.id, _segments_str(p), p.judgment.get("hook"),
                        p.judgment.get("cliffhanger"), "", "", ""])
    files["editor_review_sheet.csv"] = str(out / "editor_review_sheet.csv")

    (out / "report.md").write_text(_report_md(res, idx, title), encoding="utf-8")
    files["report.md"] = str(out / "report.md")
    return files


def write_status(out_dir: str | Path, complete: bool, model: str, pending_episodes: list,
                 pending_clips: list, stop_reason: str, rerun: str) -> Path:
    """run_status.json: the one file that says whether this output folder is finished."""
    import time
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = {"complete": complete, "model": model, "pending_episodes": pending_episodes,
              "pending_clips": pending_clips, "stop_reason": stop_reason,
              "rerun_to_continue": None if complete else rerun,
              "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return out / "run_status.json"


def _report_md(res: RunResult, idx: Index, title: str) -> str:
    s = res.stats
    L = [f"# Promo finder report: {title}".rstrip(": "), "",
         f"Mode: **{s['mode']}**" + (f" ({s['llm']['provider']} / {s['llm']['model']})" if "llm" in s else ""),
         f"Episodes: {s['episodes']} | moments: {s['n_moments']} | candidate clips: {s['n_candidates']} "
         f"| judged: {s['n_judged']} | ranked: {s['n_ranked']} | recommended: {s['n_recommended']}", ""]
    if not s.get("complete", True):
        L += [f"> **PARTIAL RESULTS.** Episodes with no model result: "
              f"{s['moments']['episodes_pending'] or 'none'}; clips not judged: {len(s['judge']['pending'])}. "
              f"Nothing was filled in for them. Re-run the same command to finish.", ""]
    if "llm" in s:
        u = s["llm"]
        L += [f"API calls (all runs of this output folder): {u['calls']}, retries {u['retries']}, "
              f"tokens in/out: {u['input_tokens']:,}/{u['output_tokens']:,}, model time {u['seconds']:.0f}s", ""]
    if "moments" in s:
        m = s["moments"]
        L += [f"Model-returned moments: {m['returned']}; dropped as unknown ID: {m['invalid_ids']}, "
              f"wrong episode: {m['wrong_episode_ids']}, bad fields: {m['invalid_fields']}", ""]

    L += ["## Post first (top 10)", ""]
    for p in res.ranked[:10]:
        j = p.judgment
        L += [f"### #{p.rank} {p.id} [{p.status}] score {p.score:.2f} | {p.duration:.0f}s | {p.kind}",
              f"- Segments: {_segments_str(p)}",
              f"- Hook: {j.get('hook')}",
              f"- Cliffhanger: {j.get('cliffhanger')}",
              f"- Why: {j.get('why')}",
              f"- Scores (hook/cliff/standalone): {j.get('hook_score')}/{j.get('cliffhanger_score')}/"
              f"{j.get('standalone_score')}"
              + (f"; issues: {'; '.join(j['issues'])}" if j.get("issues") else ""), ""]
        L += ["  <details><summary>clip transcript</summary>", ""]
        for t, ev, _ in clip_events(p, idx):
            txt = f'{ev.label()}: "{ev.text}"' if ev.type == "dialogue" else f"{ev.type.upper()}: {ev.text}"
            L.append(f"  - `{t:5.1f}s` {txt}")
        L += ["", "  </details>", ""]

    L += ["## Full ranking", "", "| # | id | status | segments | dur | hook | cliffhanger | H/C/S |",
          "|---|---|---|---|---|---|---|---|"]
    for p in res.ranked:
        j = p.judgment
        cell = lambda x: str(x or "").replace("|", "/")[:90]
        L.append(f"| {p.rank} | {p.id} | {p.status} | {_segments_str(p)} | {p.duration:.0f}s | "
                 f"{cell(j.get('hook'))} | {cell(j.get('cliffhanger'))} | "
                 f"{j.get('hook_score')}/{j.get('cliffhanger_score')}/{j.get('standalone_score')} |")

    rejected = [p for p in res.candidates if p.status == "rejected"]
    if rejected:
        L += ["", "## Rejected by hard checks", ""]
        for p in rejected:
            L.append(f"- {p.id} {_segments_str(p)}: {_failed(p)}")
    return "\n".join(L) + "\n"
