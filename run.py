#!/usr/bin/env python3
"""Find promo clips in a microdrama screenplay pack.

  python run.py --input pack.zip --out outputs/rules --provider rules
  python run.py --check --provider gemini --model gemini-3.8-flash      # key + model test, no episodes
  python run.py --input pack.zip --out outputs/gemini --provider gemini --model gemini-3.8-flash
  python run.py --input pack/ --out outputs/x --provider custom --base-url https://host/v1 --model m

Exit codes: 0 complete | 2 incomplete or API unavailable (finished work is saved; rerun the
same command to continue) | 1 error that a rerun will not fix (key, model name, quota, input).
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from promo.llm import FatalError, LLMError, ServiceUnavailable, make_llm
from promo.parse import load_episodes
from promo.pipeline import Index
from promo.pipeline import run as run_rules
from promo.report import write_outputs, write_status
from promo.runner import preflight, run_llm


def resolve_input(path: str) -> tuple[Path, str]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="promo_pack_"))
        with zipfile.ZipFile(p) as z:
            z.extractall(tmp)
        dirs = [d for d in tmp.iterdir() if d.is_dir()]
        return tmp, (dirs[0].name if len(dirs) == 1 else p.stem)
    if p.is_dir():
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        series = subdirs[0].name if len(subdirs) == 1 and not list(p.glob("*.txt")) else p.name
        return p, series
    raise SystemExit(f"error: input not found or not a .zip/folder: {path}")


def rerun_hint(argv: list[str]) -> str:
    """The same command, minus --fresh (which would wipe progress) and with the key hidden."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--fresh":
            continue
        if a == "--api-key":
            out += [a, "<your-key>"]
            skip = True
            continue
        out.append("--api-key=<your-key>" if a.startswith("--api-key=") else a)
    return "python " + shlex.join(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = ap.add_argument_group("input / output")
    io.add_argument("--input", help="zip or folder of episode .txt screenplays")
    io.add_argument("--out", default="outputs/run", help="output folder (also holds resumable state)")
    io.add_argument("--series", help="series title (default: folder name)")
    m = ap.add_argument_group("model")
    m.add_argument("--provider", default="rules",
                   help="rules | gemini | groq | openrouter | openai | custom (OpenAI-compatible)")
    m.add_argument("--model", help="model name (gemini default: gemini-3.8-flash)")
    m.add_argument("--fallback-models", default="",
                   help="comma list tried in order if --model fails the test request; the first that "
                        "answers is used for the whole run")
    m.add_argument("--base-url", help="API base URL (required for --provider custom)")
    m.add_argument("--api-key", help="API key (prefer the env var GEMINI_API_KEY / PROMO_API_KEY)")
    m.add_argument("--thinking", choices=["low", "medium", "high"],
                   help="Gemini 3+ thinking level (default: the model's own default)")
    m.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature (default: not sent; Gemini 3+ docs say leave it unset)")
    r = ap.add_argument_group("reliability")
    r.add_argument("--concurrency", type=int, default=1, help="max API requests in flight (default 1)")
    r.add_argument("--rpm", type=float, default=5, help="max request starts per minute (default 5)")
    r.add_argument("--max-retries", type=int, default=8,
                   help="retries per request on 429/5xx/network, exponential backoff 5s..120s (default 8)")
    r.add_argument("--timeout", type=int, default=300, help="seconds to wait for one response (default 300)")
    r.add_argument("--fresh", action="store_true", help="discard saved progress in --out and start over")
    r.add_argument("--allow-partial", action="store_true",
                   help="write outputs even if some episodes/clips have no model result (marked PARTIAL)")
    r.add_argument("--skip-test", action="store_true", help="skip the test request before the run")
    r.add_argument("--check", action="store_true",
                   help="only test the key, endpoint and model (lists models, one tiny request)")
    p = ap.add_argument_group("pipeline")
    p.add_argument("--episodes-per-call", type=int, default=5, help="episodes per moment-finding request")
    p.add_argument("--judge-top-k", type=int, default=60, help="how many clips the model judges")
    p.add_argument("--judge-batch", type=int, default=12, help="clips per judge request")
    p.add_argument("--early-bias", type=float, default=0.10,
                   help="score penalty spread from first to last episode (0 = off)")
    return ap


def load(a) -> tuple[list, str]:
    if not a.input:
        raise SystemExit("error: --input is required")
    folder, series = resolve_input(a.input)
    episodes = load_episodes(folder)
    if not episodes:
        raise SystemExit("error: no episodes parsed - expected .txt screenplays with timed blocks")
    warn = sum(len(e.warnings) for e in episodes)
    print(f"parsed {len(episodes)} episodes ({sum(len(e.events) for e in episodes)} events, "
          f"{warn} parse warnings) - series: {a.series or series}")
    return episodes, a.series or series


def settings(a) -> dict:
    return {k: v for k, v in vars(a).items() if k != "api_key"}


def do_check(llm, fallbacks: list[str]) -> int:
    if hasattr(llm, "endpoint"):
        print(f"endpoint: POST {llm.endpoint()}")
    try:
        models = llm.list_models()
    except LLMError as e:
        print(f"could not list models: {e}")
        models = None
    if models is not None:
        flash = sorted(m for m in models if "flash" in m.lower())
        print(f"{len(models)} models can generate content with this key")
        if flash:
            print("flash models: " + ", ".join(flash))
        for m in [llm.model, *fallbacks]:
            print(f"  {m}: {'listed' if m in models else 'NOT LISTED for this key'}")
    print("test request:")
    try:
        chosen = preflight(llm, fallbacks, print)
    except FatalError as e:
        print(f"check failed: {e}")
        return 1
    except ServiceUnavailable as e:
        print(f"check failed - the API is not answering right now:\n  {e}\n"
              f"HTTP 503 'overloaded' and 429 are on the provider's side: your key, endpoint and model "
              f"name are fine. Try again later, or pass --fallback-models with another model from the "
              f"list above.")
        return 2
    print(f"OK: {chosen} answered. Run with --model {chosen}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv) if argv is None else ["run.py", *argv]
    a = build_parser().parse_args(argv[1:])
    fallbacks = [m.strip() for m in a.fallback_models.split(",") if m.strip()]

    if a.provider == "rules":
        if a.check:
            raise SystemExit("error: --check needs an API provider, e.g. --provider gemini")
        episodes, series = load(a)
        t0 = time.time()
        res = run_rules(episodes, series=series, early_bias=a.early_bias)
        res.stats["wall_seconds"] = round(time.time() - t0, 1)
        res.stats["settings"] = settings(a)
        files = write_outputs(res, Index(episodes), a.out, title=series)
        print(json.dumps({k: v for k, v in res.stats.items() if k != "settings"}, indent=2))
        print("wrote:", ", ".join(sorted(files)), "->", a.out)
        return 0

    try:
        llm = make_llm(a.provider, model=a.model, base_url=a.base_url, api_key=a.api_key,
                       thinking=a.thinking, rpm=a.rpm, max_retries=a.max_retries, timeout=a.timeout,
                       temperature=a.temperature)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if a.check:
        return do_check(llm, fallbacks)

    episodes, series = load(a)
    hint = rerun_hint(argv)
    t0 = time.time()
    try:
        r = run_llm(episodes, llm, a.out, series=series, fallback_models=fallbacks, fresh=a.fresh,
                    allow_partial=a.allow_partial, concurrency=a.concurrency,
                    episodes_per_call=a.episodes_per_call, judge_batch=a.judge_batch,
                    judge_top_k=a.judge_top_k, early_bias=a.early_bias, test_request=not a.skip_test)
    except FatalError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    except ServiceUnavailable as e:
        print(f"\nThe API is not answering right now, so no new work was sent:\n  {e}\n"
              f"Saved progress is untouched. Try again later:\n  {hint}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"\nstopped. Finished episodes/clips are saved; rerun to continue:\n  {hint}")
        return 130

    r.stats["wall_seconds"] = round(time.time() - t0, 1)
    r.stats["settings"] = settings(a)
    write_status(a.out, r.complete, r.stats["llm"]["model"], r.pending_episodes, r.pending_clips,
                 r.stop_reason, hint)
    if r.result is not None:
        write_outputs(r.result, Index(episodes), a.out, title=series)
    u = r.stats["llm"]
    print(f"\nmodel {u['model']}: {u['calls']} calls, {u['retries']} retries, "
          f"{u['input_tokens']:,} in / {u['output_tokens']:,} out tokens (all runs of this folder)")
    if r.complete:
        print(f"COMPLETE: {r.stats['n_ranked']} promos ranked ({r.stats['n_recommended']} recommended) "
              f"-> {a.out}/promos_ranked.csv, {a.out}/report.md")
        return 0
    print(f"\nINCOMPLETE: {len(r.pending_episodes)} episode(s) and {len(r.pending_clips)} clip(s) have no "
          f"model result yet. Reason: {r.stop_reason}")
    if r.pending_episodes:
        print(f"  pending episodes: {', '.join(map(str, r.pending_episodes))}")
    print("Nothing was filled in for them" + (" (outputs are marked PARTIAL)." if r.result else
                                                ", and no promo files were written."))
    print(f"Finished work is saved in {a.out}/state. Rerun the same command to continue:\n  {hint}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
