# Promo Finder

Reads a microdrama screenplay pack (one timestamped `.txt` per episode) and returns ranked
20-60 second promo clips. Each clip comes with its segments (episode + start/end), the hook,
the cliffhanger, one line on why it works, scores, and the results of automatic quality checks.

Built for Kuku's AI Generalist assignment, Problem 1. The input pack is confidential and is
**not** in this repo; `sample/` holds a short synthetic show written for tests and the demo.

## How it works

```
screenplays ──► 1. parse ──► 2. moments ──► 3. assemble ──► 4. judge ──► 5. checks + rank ──► outputs
               (code)       (LLM/rules)     (code)          (LLM/rules)   (code)
```

1. **Parse** each block (camera, action, SFX, dialogue) into an event with an ID and exact times.
2. **Moments**: the model reads 5 episodes per request and names hook beats and cliffhanger beats
   **by event ID only**, with a 1-5 strength.
3. **Assemble** (code) pairs hooks with later cliffhangers into clips: one continuous stretch if it
   fits 20-60s, otherwise two segments from one episode or across consecutive episodes. Starts snap
   to the camera cut that carries the hook; ends cut right after the cliffhanger beat and before the
   next line, so the answer never leaks in.
4. **Judge**: the model sees each clip exactly as a cold viewer would and scores hook (first 3s),
   cliffhanger, and standalone sense, and writes the hook / cliffhanger / why lines.
5. **Checks + rank** (code): hard checks (20-60s, hook within 3s, ends on the cliff beat, nothing
   answers it before the cut, valid segments), then overlap dedupe, one promo per opening beat,
   max 2 per episode, and a small bias toward early episodes.

The model only ever returns IDs and scores; every timestamp comes from the script, so a model can't
invent a time. Unknown IDs are dropped and counted (`run_stats.json`).

`--provider rules` runs the whole pipeline with keyword/heuristic scorers and no API. It is the
baseline the LLM version is compared against. It is never mixed into an LLM run.

## Run it

Python 3.9+.

```bash
pip install -r requirements.txt
python -m pytest -q                                   # offline tests, no key needed

# baseline, no API
python run.py --input path/to/pack.zip --out outputs/rules --provider rules

# Gemini
export GEMINI_API_KEY="your-key"                      # PowerShell: $env:GEMINI_API_KEY="your-key"
python run.py --check --provider gemini --model gemini-3.8-flash
python run.py --input path/to/pack.zip --out outputs/gemini --provider gemini \
    --model gemini-3.8-flash --fallback-models gemini-3.5-flash
```

About 16 requests for 50 episodes (1 test request, 10 moment requests, 5 judge requests).

Exit codes: `0` complete · `2` incomplete or API unavailable (finished work is saved; rerun the
same command) · `1` an error a rerun won't fix (key, model name, daily quota, input).

## When the API fails

| What happens | What the tool does |
|---|---|
| Before any episode is sent | One tiny test request checks the key, endpoint and model. If it fails, nothing else is sent. |
| `--model` fails the test request | Tries each `--fallback-models` entry the key can see, in order. The one that answers is **pinned** for the whole run (never mixed). |
| HTTP 503 / 500 / 502 / 504 / 408, network error, timeout | Retried with exponential backoff and jitter: about 5s, 10s, 20s, 40s, 80s, then 120s, up to `--max-retries` (8). Uses the server's `retryDelay` if longer. All workers pause together. |
| HTTP 429 per-minute rate limit | Retried the same way. Requests are also paced to `--rpm` (5/min). |
| HTTP 429 daily quota, or free-tier `limit: 0` | Stops immediately (retrying won't help). |
| Still failing after all retries | Stops sending new requests. Already-finished work is kept. Exit code 2. |
| Bad key (401/403/400), unknown model (404) | Stops immediately with the reason. |
| Reply blocked, cut off (`MAX_TOKENS`), empty, or invalid JSON twice | That request's episodes/clips stay pending and are retried in smaller requests. |
| Reply has unknown event IDs, missing episodes, or scores outside 1-5 | Those items are dropped, never guessed. |
| Ctrl-C | Requests in flight finish; everything saved so far stays saved. |

**No made-up results.** A result is saved only from a valid model reply. Nothing is filled in by
rules, defaults or guesses. If anything is still pending at the end, no promo files are written
(unless you pass `--allow-partial`, which writes them marked PARTIAL), and `run_status.json` lists
what is pending.

**Resumable.** Every finished episode and every judged clip is its own file in `<out>/state/`.
Rerunning the same command skips them and sends only what's left; a finished run makes no API calls.
`--fresh` discards saved progress. A different pack, prompt, or model in the same `--out` is refused
rather than mixed.

**Concurrency.** `--concurrency 1` (default) sends one request at a time; raise it to 2-4 on a paid
tier. The `--rpm` pacing and the backoff pause are shared by all workers.

## Swap the model

| Provider | Flag | Key env var |
|---|---|---|
| Gemini (native API, works with `AQ.` keys) | `--provider gemini --model gemini-3.8-flash` | `GEMINI_API_KEY` |
| Groq | `--provider groq --model llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| OpenRouter | `--provider openrouter --model <id>` | `OPENROUTER_API_KEY` |
| OpenAI | `--provider openai --model <id>` | `OPENAI_API_KEY` |
| Any OpenAI-compatible endpoint (internal gateway, vLLM, Ollama, LiteLLM, Anthropic compat) | `--provider custom --base-url https://host/v1 --model <id>` | `PROMO_API_KEY` |

Gemini endpoint: `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`,
key in the `x-goog-api-key` header. `temperature` is not sent unless you pass `--temperature`
(Gemini 3+ docs say to leave sampling parameters unset). `--thinking low|medium|high` sets the
Gemini 3+ thinking level; leave it unset for the model default.

Other knobs: `--judge-top-k 60` (clips the model judges), `--early-bias 0.10` (0 turns off the
preference for early episodes), `--episodes-per-call 5`, `--timeout 300`.

## Outputs (`--out` folder)

| File | What |
|---|---|
| `promos_ranked.csv` / `.json` | Final ranked promos: segments, hook, cliffhanger, why, scores, checks |
| `report.md` | Top 10 with the clip transcript, full ranking, rejected clips |
| `editor_review_sheet.csv` | For the human pass: post / fix / reject per promo |
| `all_candidates.json` | Every candidate clip, including rejected and not-judged ones |
| `moments.json` | Hook and cliffhanger beats found per episode |
| `run_stats.json` | Counts, model, calls, tokens (across reruns), dropped items, settings |
| `run_status.json` | Complete or not, what is pending, the command to continue |
| `state/` | Saved progress (one file per episode / judged clip), usage and failure logs |

## Web app

```bash
streamlit run app.py
```

Upload a `.zip` (or the `.txt` files), pick a provider, paste a key (not stored), run, download
the outputs. If the API fails partway, click **Find promos** again to continue. To deploy: push
this repo to GitHub, then on share.streamlit.io create an app from `app.py`. Optionally add
`PROMO_API_KEY` under the app's Secrets so viewers don't need a key.

## Input format

```
EPISODE NUMBER: Episode 1
EPISODE LOGLINE: ...
--- SCRIPT START ---
SCENE 01
Scene start: 00:00.00
**CAMERA:**
Start: (00:00.00) | End: (00:04.80)
Close-up on ...
**LADY JULIANA (Confused, Breathless):**
Start: (00:13.50) | End: (00:14.30)
"Why?"
--- SCRIPT END ---
```

## Known limits

- It reads the script, not the video: it cannot see acting, music or how a shot actually looks.
- No "flash-forward" structure (show the climax first, then rewind); clips play in story order.
- Cross-episode clips assume consecutive episodes continue the same scene.
- The rules baseline matches words, not meaning (it scores "strikes flint" as violence).
