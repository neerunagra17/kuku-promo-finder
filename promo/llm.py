"""Model providers. Swapping models = changing provider / model / base_url, nothing else.

Providers
  gemini   Google Gemini REST API:
           POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
           key in the x-goog-api-key header (works for new 'AQ.' keys and old 'AIza' keys)
  groq / openrouter / openai / custom
           any OpenAI-compatible POST {base_url}/chat/completions
  mock     offline stand-in used by the tests

Error handling (nothing is ever invented to fill a failed request)
  429 per-minute, 408, 5xx, network, timeout
        -> retried with exponential backoff + jitter (5s, 10s, 20s ... capped at 120s, or the
           server's retryDelay if longer). Every worker pauses together, not just the one that
           failed. After max_retries -> ServiceUnavailable: the run stops, finished work is kept.
  429 daily quota / free-tier "limit: 0"   -> FatalError(kind="quota"), not retried
  401 / 403 / 400 "API key not valid"      -> FatalError(kind="auth")
  404 (unknown model or endpoint)          -> FatalError(kind="model")
  other 4xx                                -> FatalError(kind="request")
  blocked prompt, output cut off, no text, invalid JSON twice
        -> UnitError: this one request is unusable; nothing is saved for it
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field

import requests

PRESETS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta",
               "model": "gemini-3.8-flash", "key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"]},
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "model": "llama-3.3-70b-versatile", "key_env": ["GROQ_API_KEY"]},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "model": "meta-llama/llama-3.3-70b-instruct:free", "key_env": ["OPENROUTER_API_KEY"]},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini",
               "key_env": ["OPENAI_API_KEY"]},
}

REPAIR = "\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON object."


class LLMError(RuntimeError):
    """Base class for model errors."""


class FatalError(LLMError):
    """Retrying will not help (bad key, unknown model, daily quota, bad request, mismatched state)."""

    def __init__(self, msg: str, kind: str = "request"):
        super().__init__(msg)
        self.kind = kind


class ServiceUnavailable(LLMError):
    """Still 429/5xx/network after all retries. Stop; finished work is kept; rerun later."""


class UnitError(LLMError):
    """This one request's output is unusable. Nothing is saved for it; it stays pending."""


class Stopped(LLMError):
    """The run is stopping, so this request was not sent."""


class TransientError(Exception):
    """Internal: a failure worth retrying."""

    def __init__(self, msg: str, retry_after: float | None = None, status: int | None = None):
        super().__init__(msg)
        self.retry_after = retry_after
        self.status = status


@dataclass
class Reply:
    data: dict = field(default_factory=dict)
    calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


@dataclass
class Usage:
    calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    notes: list = field(default_factory=list)


class RateLimiter:
    """Thread-safe pacing shared by all workers.

    acquire(): at most `rpm` request starts per minute.
    pause(s):  every worker waits s seconds (used after a 429/503 so they back off together)."""

    def __init__(self, rpm: float = 0):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self, stop: threading.Event | None = None):
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next:
                    self._next = now + self.interval
                    return
                wait = self._next - now
            if stop is not None:
                if stop.wait(min(wait, 1.0)):
                    raise Stopped("run is stopping")
            else:
                time.sleep(min(wait, 1.0))

    def pause(self, seconds: float):
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


class BaseLLM:
    name = "base"

    def __init__(self, model: str, rpm: float = 5, max_retries: int = 8, timeout: int = 300,
                 temperature: float | None = None, backoff_base: float = 5.0, backoff_cap: float = 120.0,
                 log=print):
        self.model = model
        self.limiter = RateLimiter(rpm)
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature      # None = model default (Gemini 3+ asks you not to set it)
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.log = log
        self.usage = Usage()
        self.stop_event = threading.Event()
        self.on_call = None                 # callback(**row) for every API attempt (usage log)
        self._lock = threading.Lock()

    # -- public -----------------------------------------------------------------------
    def backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Exponential backoff with jitter: base * 2^attempt, capped, randomised to 50-100%.
        Never shorter than the server's own retry hint."""
        ceiling = min(self.backoff_cap, self.backoff_base * (2 ** attempt))
        return max(random.uniform(ceiling / 2, ceiling), retry_after or 0.0)

    def generate_json(self, system: str, user: str, meta: dict | None = None,
                      max_retries: int | None = None) -> Reply:
        """One logical request -> parsed JSON object. Raises instead of guessing."""
        rep = Reply()
        text = self._request(system, user, meta, rep, max_retries)
        try:
            data = parse_json(text)
        except ValueError:
            self._note("invalid JSON; asked the model to repair it")
            text = self._request(system, user + REPAIR, meta, rep, max_retries)
            try:
                data = parse_json(text)
            except ValueError as e:
                raise UnitError("model returned invalid JSON twice") from e
        if not isinstance(data, dict):
            raise UnitError("model returned JSON that is not an object")
        rep.data = data
        return rep

    def list_models(self) -> list[str] | None:
        return None

    # -- plumbing ---------------------------------------------------------------------
    def _request(self, system, user, meta, rep: Reply, max_retries) -> str:
        retries = self.max_retries if max_retries is None else max_retries
        task = (meta or {}).get("task", "")
        for attempt in range(retries + 1):
            if self.stop_event.is_set():
                raise Stopped("run is stopping")
            self.limiter.acquire(self.stop_event)
            t0 = time.monotonic()
            try:
                text, n_in, n_out = self._call(system, user, meta)
            except TransientError as e:
                rep.retries += 1
                self._add(retries=1)
                self._emit(task=task, model=self.model, retries=1, error=str(e)[:200])
                if attempt >= retries:
                    raise ServiceUnavailable(f"{self.model}: {e} (gave up after {attempt + 1} attempts)") from e
                wait = self.backoff_delay(attempt, e.retry_after)
                self.limiter.pause(wait)
                self.log(f"  [{self.model}] {e} - retry {attempt + 1}/{retries} in {wait:.0f}s")
                if self.stop_event.wait(wait):
                    raise Stopped("run is stopping")
                continue
            dt = time.monotonic() - t0
            rep.calls += 1
            rep.input_tokens += n_in
            rep.output_tokens += n_out
            rep.seconds += dt
            self._add(calls=1, input_tokens=n_in, output_tokens=n_out, seconds=dt)
            self._emit(task=task, model=self.model, calls=1, input_tokens=n_in, output_tokens=n_out,
                       seconds=round(dt, 2))
            return text
        raise AssertionError("unreachable")

    def _call(self, system, user, meta) -> tuple[str, int, int]:  # pragma: no cover
        raise NotImplementedError

    def _add(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self.usage, k, getattr(self.usage, k) + v)

    def _note(self, msg):
        with self._lock:
            self.usage.notes.append(msg)

    def _emit(self, **row):
        if self.on_call:
            self.on_call(**row)


# ----------------------------------------------------------------------------- HTTP errors
DAILY_OR_ZERO = re.compile(r"PerDay|per day|requests per day|tokens per day|\bRPD\b|\bTPD\b|limit:\s*0\b",
                           re.I)


def _error_message(r) -> str:
    try:
        err = r.json().get("error", {})
        if isinstance(err, dict):
            return str(err.get("message") or err.get("status") or "")[:300]
        return str(err)[:300]
    except (ValueError, AttributeError):
        return (r.text or "")[:300]


def _retry_after(r) -> float | None:
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', r.text or "")
    if m:
        return float(m.group(1))
    try:
        return float(r.headers.get("retry-after", ""))
    except (TypeError, ValueError):
        return None


def _raise_for_status(r, model: str):
    code = r.status_code
    if code < 400:
        return
    msg = _error_message(r)
    if code == 429:
        if DAILY_OR_ZERO.search(r.text or ""):
            raise FatalError(f"{model}: quota used up for today, or this model has no free-tier quota for "
                             f"your key (HTTP 429: {msg}). Wait for the daily reset, pick another model, "
                             f"or enable billing.", kind="quota")
        raise TransientError(f"HTTP 429 rate limited: {msg}", _retry_after(r), 429)
    if code in (408, 500, 502, 503, 504):
        raise TransientError(f"HTTP {code} {msg}".strip(), _retry_after(r), code)
    if code in (401, 403) or (code == 400 and "api key" in msg.lower()):
        raise FatalError(f"API key rejected (HTTP {code}: {msg})", kind="auth")
    if code == 404:
        raise FatalError(f"model '{model}' or endpoint not found (HTTP 404: {msg}). Run "
                         f"`python run.py --check --provider ...` to list the models your key can use.",
                         kind="model")
    raise FatalError(f"{model}: HTTP {code}: {msg}", kind="request")


def normalize_model(m: str | None) -> str:
    m = (m or "").strip()
    return m[len("models/"):] if m.startswith("models/") else m


# ------------------------------------------------------------------------------- providers
class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, api_key: str, model: str, base_url: str | None = None,
                 thinking: str | None = None, **kw):
        super().__init__(normalize_model(model), **kw)
        self.api_key = api_key
        self.base_url = (base_url or PRESETS["gemini"]["base_url"]).rstrip("/")
        self.thinking = thinking

    def endpoint(self) -> str:
        return f"{self.base_url}/models/{self.model}:generateContent"

    def request_body(self, system: str, user: str) -> dict:
        gen: dict = {"responseMimeType": "application/json"}
        if self.temperature is not None:
            gen["temperature"] = self.temperature
        if self.thinking:
            gen["thinkingConfig"] = {"thinkingLevel": self.thinking}   # Gemini 3+: low | medium | high
        return {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": gen}

    def _call(self, system, user, meta):
        try:
            r = requests.post(self.endpoint(), json=self.request_body(system, user),
                              headers={"x-goog-api-key": self.api_key}, timeout=(15, self.timeout))
        except requests.RequestException as e:
            raise TransientError(f"network error ({type(e).__name__})")
        _raise_for_status(r, self.model)
        try:
            data = r.json()
        except ValueError:
            raise TransientError("response was not JSON")
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise UnitError(f"prompt blocked by Gemini ({block})")
        cands = data.get("candidates") or []
        if not cands:
            raise TransientError("empty response (no candidates)")
        c = cands[0]
        finish = c.get("finishReason", "STOP")
        parts = (c.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if finish not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            raise UnitError(f"response ended early (finishReason={finish})")
        if not text.strip():
            raise UnitError("model returned no text")
        um = data.get("usageMetadata") or {}
        return (text, um.get("promptTokenCount", 0),
                um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0))

    def list_models(self) -> list[str]:
        names, token = [], None
        for _page in range(20):
            params = {"pageSize": 1000}
            if token:
                params["pageToken"] = token
            for attempt in range(4):
                try:
                    r = requests.get(f"{self.base_url}/models", headers={"x-goog-api-key": self.api_key},
                                     params=params, timeout=(15, 60))
                    _raise_for_status(r, self.model)
                    break
                except (requests.RequestException, TransientError) as e:
                    if attempt == 3:
                        raise ServiceUnavailable(f"could not list models: {e}")
                    time.sleep(self.backoff_delay(attempt))
            data = r.json()
            for m in data.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    names.append(normalize_model(m.get("name", "")))
            token = data.get("nextPageToken")
            if not token:
                break
        return names


class OpenAICompatLLM(BaseLLM):
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str, json_mode: bool = True, **kw):
        super().__init__(model, **kw)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.json_mode = json_mode

    def _call(self, system, user, meta):
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=(15, self.timeout),
                              headers={"Authorization": f"Bearer {self.api_key}"})
        except requests.RequestException as e:
            raise TransientError(f"network error ({type(e).__name__})")
        if r.status_code == 400 and self.json_mode and "response_format" in (r.text or ""):
            self.json_mode = False          # some servers reject JSON mode; the prompt still asks for JSON
            return self._call(system, user, meta)
        _raise_for_status(r, self.model)
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise UnitError("response ended early (finish_reason=length)")
        text = (choice.get("message") or {}).get("content") or ""
        if not text.strip():
            raise UnitError("model returned no text")
        u = data.get("usage") or {}
        return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    def list_models(self) -> list[str] | None:
        try:
            r = requests.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"},
                             timeout=(15, 60))
        except requests.RequestException:
            return None
        if r.status_code in (401, 403):
            _raise_for_status(r, self.model)
        if r.status_code >= 400:
            return None
        return [m.get("id") for m in r.json().get("data", [])]


class MockLLM(BaseLLM):
    """Offline provider for tests. responder(meta, model) returns a dict/str or raises
    TransientError / FatalError / UnitError to simulate API behaviour."""
    name = "mock"

    def __init__(self, responder, model: str = "mock-1", models: list[str] | None = None, **kw):
        kw.setdefault("rpm", 0)
        kw.setdefault("backoff_base", 0.0)
        kw.setdefault("log", lambda *a, **k: None)
        super().__init__(model, **kw)
        self.responder = responder
        self.models = models

    def _call(self, system, user, meta):
        out = self.responder(meta or {}, self.model)
        return (out if isinstance(out, str) else json.dumps(out)), 0, 0

    def list_models(self):
        return self.models


# --------------------------------------------------------------------------------- helpers
def parse_json(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(t[a:b + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("no JSON object in model output")


def resolve_key(provider: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    envs = ["PROMO_API_KEY"] + PRESETS.get(provider, {}).get("key_env", [])
    for env in envs:
        if os.environ.get(env):
            return os.environ[env]
    raise FatalError(f"no API key for provider '{provider}'. Set {' or '.join(envs)}.", kind="auth")


def make_llm(provider: str, model: str | None = None, base_url: str | None = None,
             api_key: str | None = None, thinking: str | None = None, **kw) -> BaseLLM:
    """provider: gemini | groq | openrouter | openai | custom (OpenAI-compatible, needs base_url)."""
    provider = provider.lower()
    if provider == "gemini":
        return GeminiLLM(resolve_key("gemini", api_key), model or PRESETS["gemini"]["model"],
                         base_url=base_url, thinking=thinking, **kw)
    if thinking:
        raise FatalError("--thinking only applies to --provider gemini", kind="request")
    if provider in PRESETS or provider == "custom":
        preset = PRESETS.get(provider, {})
        url = base_url or preset.get("base_url")
        if not url:
            raise FatalError("--base-url is required for provider 'custom'", kind="request")
        if not (model or preset.get("model")):
            raise FatalError("--model is required for provider 'custom'", kind="request")
        return OpenAICompatLLM(resolve_key(provider, api_key), model or preset["model"], url, **kw)
    raise FatalError(f"unknown provider '{provider}'", kind="request")
