"""Streamlit UI: upload screenplays -> ranked promos.   Run: streamlit run app.py"""
from __future__ import annotations

import io
import os
import tempfile
import threading
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from promo.llm import PRESETS, FatalError, LLMError, ServiceUnavailable, make_llm
from promo.parse import fmt_ts, load_episodes
from promo.pipeline import Index, clip_events
from promo.pipeline import run as run_rules
from promo.report import write_outputs
from promo.runner import run_llm

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "sample"

st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Promo Finder")
st.caption("Reads a microdrama screenplay pack and returns ranked 20-60s promo clips: "
           "segments with timestamps, the hook, the cliffhanger and why it works.")

with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", ["gemini", "rules (no API)", "groq", "openrouter", "openai",
                                         "custom (OpenAI-compatible)"])
    pkey = provider.split(" ")[0]
    model = base_url = api_key = None
    fallbacks: list[str] = []
    if pkey != "rules":
        model = st.text_input("Model", PRESETS.get(pkey, {}).get("model", ""))
        fallbacks = [m.strip() for m in st.text_input("Fallback models (comma separated)", "").split(",")
                     if m.strip()]
        if pkey == "custom":
            base_url = st.text_input("Base URL", "https://")
        default_key = ""
        try:
            default_key = st.secrets.get("PROMO_API_KEY", "")
        except Exception:  # noqa: BLE001 - no secrets file
            pass
        api_key = st.text_input("API key (not stored)", value="", type="password",
                                help="Leave empty to use the PROMO_API_KEY secret / env var") \
            or default_key or os.environ.get("PROMO_API_KEY", "")
        rpm = st.number_input("Requests per minute", 1, 600, 5)
        concurrency = st.number_input("Requests in flight", 1, 4, 1)
        max_retries = st.number_input("Retries per request (429/5xx)", 0, 12, 6)
        partial = st.checkbox("Show partial results if some requests fail", False)
    st.header("Ranking")
    early_bias = st.slider("Prefer early episodes", 0.0, 0.3, 0.10, 0.01,
                           help="Penalty spread from first to last episode. Promos from early episodes "
                                "pay off sooner for a new viewer.")
    top_k = st.slider("Clips the model judges", 10, 150, 60, 10)

src = st.radio("Input", ["Upload .zip or .txt files", "Synthetic sample show (2 episodes)"], horizontal=True)
files = None
if src.startswith("Upload"):
    files = st.file_uploader("Episode screenplays", type=["zip", "txt"], accept_multiple_files=True)

if "work" not in st.session_state:
    st.session_state["work"] = tempfile.mkdtemp(prefix="promo_")
WORK = Path(st.session_state["work"])

if st.button("Find promos", type="primary"):
    series = "sample show"
    if src.startswith("Synthetic"):
        folder = SAMPLE
    else:
        if not files:
            st.error("Upload a .zip of episodes or the episode .txt files.")
            st.stop()
        folder = Path(tempfile.mkdtemp(prefix="pack_", dir=WORK))
        for f in files:
            if f.name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                    z.extractall(folder)
                series = Path(f.name).stem
            else:
                (folder / f.name).write_bytes(f.getvalue())
        dirs = [d for d in folder.iterdir() if d.is_dir()]
        if len(dirs) == 1:
            series = dirs[0].name
    episodes = load_episodes(folder)
    if not episodes:
        st.error("No episodes parsed. Expected .txt screenplays with timed blocks (see README).")
        st.stop()
    st.write(f"Parsed **{len(episodes)}** episodes, {sum(len(e.events) for e in episodes):,} timed events.")
    idx = Index(episodes)
    out = WORK / "outputs" / pkey

    log_box = st.empty()
    lines: list[str] = []
    ui_thread = threading.get_ident()

    def log(msg):
        lines.append(str(msg))
        if threading.get_ident() == ui_thread:      # worker threads can't draw; they show on next update
            log_box.code("\n".join(lines[-14:]))

    if pkey == "rules":
        res = run_rules(episodes, series=series, early_bias=early_bias, log=log)
    else:
        try:
            llm = make_llm(pkey, model=model or None, base_url=base_url, api_key=api_key or None,
                           rpm=rpm, max_retries=int(max_retries), log=log)
        except LLMError as e:
            st.error(str(e))
            st.stop()

        def go(fresh: bool):
            return run_llm(episodes, llm, out, series=series, fallback_models=fallbacks, fresh=fresh,
                           allow_partial=partial, concurrency=int(concurrency), judge_top_k=top_k,
                           early_bias=early_bias, log=log)
        try:
            try:
                r = go(fresh=False)
            except FatalError as e:
                if e.kind != "state":
                    raise
                r = go(fresh=True)           # new upload or new model: start a new run
        except FatalError as e:
            st.error(str(e))
            st.stop()
        except ServiceUnavailable as e:
            st.error(f"The API is not answering right now, so nothing new was sent: {e}. "
                     f"Try again in a few minutes, or add a fallback model.")
            st.stop()
        log_box.code("\n".join(lines[-14:]))
        if not r.complete:
            st.warning(f"Incomplete: {len(r.pending_episodes)} episode(s) and {len(r.pending_clips)} clip(s) "
                       f"have no model result yet ({r.stop_reason}). Nothing was filled in for them. "
                       f"Finished work is saved - click **Find promos** again to continue.")
        if r.result is None:
            st.session_state.pop("result", None)
            st.stop()
        res = r.result
    write_outputs(res, idx, out, title=series)
    st.session_state["result"] = (res, idx, out, series)

if "result" in st.session_state:
    res, idx, out, series = st.session_state["result"]
    s = res.stats
    if not s.get("complete", True):
        st.info("PARTIAL results: only episodes and clips with a real model result are shown.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate clips", s["n_candidates"])
    c2.metric("Judged", s["n_judged"])
    c3.metric("Ranked", s["n_ranked"])
    c4.metric("Recommended", s["n_recommended"])
    if "llm" in s:
        u = s["llm"]
        st.caption(f"{u['provider']} / {u['model']}: {u['calls']} calls, {u['retries']} retries, "
                   f"{u['input_tokens']:,} in / {u['output_tokens']:,} out tokens")

    rows = []
    for p in res.ranked:
        j = p.judgment
        rows.append({"#": p.rank, "status": p.status, "id": p.id,
                     "segments": " + ".join(f"Ep{x.episode} {fmt_ts(x.start)}-{fmt_ts(x.end)}" for x in p.segments),
                     "sec": p.duration, "hook": j.get("hook"), "cliffhanger": j.get("cliffhanger"),
                     "why": j.get("why"), "H/C/S": f"{j.get('hook_score')}/{j.get('cliffhanger_score')}/"
                                                   f"{j.get('standalone_score')}", "score": p.score})
    st.subheader("Ranked promos (post from the top)")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Inspect a promo")
    pick = st.selectbox("Promo", [f"#{p.rank} {p.id}" for p in res.ranked]) if res.ranked else None
    if pick:
        p = next(x for x in res.ranked if f"#{x.rank} {x.id}" == pick)
        st.write(f"**Why:** {p.judgment.get('why')}")
        if p.judgment.get("issues"):
            st.write("**Judge issues:** " + "; ".join(p.judgment["issues"]))
        failed = [f"{k} ({v['detail']})" for k, v in p.checks.items() if not v["ok"]]
        st.write("**Checks:** " + ("all passed" if not failed else "; ".join(failed)))
        txt, cur = [], 0
        for t, ev, si in clip_events(p, idx):
            if si != cur:
                txt.append(f"----- cut to episode {p.segments[si].episode} -----")
                cur = si
            body = f'{ev.label()}: "{ev.text}"' if ev.type == "dialogue" else f"{ev.type.upper()}: {ev.text}"
            txt.append(f"{t:5.1f}s  {body}")
        st.code("\n".join(txt), language=None)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(Path(out).iterdir()):
            if f.is_file():
                z.write(f, f.name)
    st.download_button("Download all outputs (.zip)", buf.getvalue(), file_name="promo_outputs.zip")
