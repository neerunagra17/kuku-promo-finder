"""Offline tests. Run: python -m pytest -q

They use the synthetic sample show in sample/, a scripted fake model and fake HTTP responses,
so they need no key, no network and no Kuku content."""
import json
import threading
import time
from pathlib import Path

import pytest

import promo.llm as L
from promo.llm import FatalError, GeminiLLM, MockLLM, ServiceUnavailable, TransientError, UnitError, parse_json
from promo.parse import load_episodes, parse_text, to_seconds
from promo.pipeline import Index, judge_rules, moments_rules
from promo.pipeline import run as run_rules
from promo.report import write_outputs
from promo.runner import _validate_judgment, run_llm

SAMPLE = Path(__file__).resolve().parent.parent / "sample"


def episodes():
    return load_episodes(SAMPLE)


def quiet(*a, **k):
    pass


# ------------------------------------------------------------------------------ parsing
def test_parse_sample():
    eps = episodes()
    assert [e.number for e in eps] == [1, 2]
    e1 = eps[0]
    assert e1.warnings == []
    assert abs(e1.duration - 76.0) < 0.01
    slap = [ev for ev in e1.events if "slaps" in ev.text]
    assert slap and slap[0].type == "action" and abs(slap[0].start - 19.2) < 1e-6
    line = next(ev for ev in e1.events if ev.speaker == "CELESTE")
    assert line.tags == "Shrill, Panicked" and line.text.startswith("My necklace")


def test_timestamp_and_label_parsing():
    assert to_seconds("01:02.50") == 62.5
    ep = parse_text('EPISODE NUMBER: Episode 9\n--- SCRIPT START ---\nSCENE 01\n'
                    '**MATRIARCH (O.S.) (Firm):**\nStart: (00:01.00) | End: (00:02.00)\n"Go."\n')
    ev = ep.events[0]
    assert ep.number == 9 and ev.speaker == "MATRIARCH (O.S.)" and ev.tags == "Firm" and ev.text == "Go."


# ------------------------------------------------------------------------ rules baseline
def test_rules_pipeline_produces_valid_promos(tmp_path):
    eps = episodes()
    res = run_rules(eps, series="sample", log=quiet)
    assert res.ranked
    for p in res.ranked:
        assert 20 <= p.duration <= 60
        hard = [k for k, v in p.checks.items() if v["level"] == "hard" and not v["ok"]]
        assert not hard, (p.id, hard)
    files = write_outputs(res, Index(eps), tmp_path, title="sample")
    assert (tmp_path / "report.md").read_text().startswith("# Promo finder report")
    assert len(files) == 7
    j = judge_rules(res.ranked[0], Index(eps))
    assert {"hook_score", "cliffhanger_score", "standalone_score", "hook", "cliffhanger"} <= set(j)


# ------------------------------------------------------------------- fake model for runs
class FakeModel:
    """Answers moments/judge requests from the rule engine; can be told to fail."""

    def __init__(self, eps, fail=None, drop_judgment=(), delay=0.0):
        self.by_no = {e.number: e for e in eps}
        self.fail = fail or (lambda meta, model: None)
        self.drop_judgment = set(drop_judgment)
        self.delay = delay
        self.requests = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def __call__(self, meta, model):
        task = meta.get("task")
        with self._lock:
            self.requests.append((task, tuple(meta.get("episodes") or meta.get("promo_ids") or ())))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            exc = self.fail(meta, model)
            if exc:
                raise exc
            if task == "preflight":
                return {"ok": True}
            if task == "moments":
                out = []
                for n in meta["episodes"]:
                    ms = moments_rules(self.by_no[n])
                    hooks = [{"id": m.event_id, "strength": m.strength, "type": "other", "why": "x"}
                             for m in ms if m.kind == "hook"]
                    hooks.append({"id": "E99-999", "strength": 5})            # invented ID
                    cliffs = [{"id": m.event_id, "strength": m.strength, "open_question": "y"}
                              for m in ms if m.kind == "cliff"]
                    out.append({"episode": n, "hooks": hooks, "cliffhangers": cliffs})
                return {"episodes": out}
            return {"judgments": [{"promo_id": i, "hook_score": 4, "cliffhanger_score": 4,
                                   "standalone_score": 4, "hook": "h", "cliffhanger": "c", "why": "w",
                                   "issues": []}
                                  for i in meta["promo_ids"] if i not in self.drop_judgment]}
        finally:
            with self._lock:
                self.in_flight -= 1

    def count(self, task):
        return sum(1 for t, _ in self.requests if t == task)

    def episodes_sent(self):
        return [n for t, unit in self.requests if t == "moments" for n in unit]


def test_llm_run_complete_uses_only_model_results(tmp_path):
    eps = episodes()
    fake = FakeModel(eps)
    r = run_llm(eps, MockLLM(fake), tmp_path, series="s", log=quiet)
    assert r.complete and r.result.ranked
    assert fake.requests[0][0] == "preflight"                      # test request goes first
    assert all(p.judgment["judge"] == "mock-1" for p in r.result.ranked)   # no rules fallback
    assert r.stats["moments"]["invalid_ids"] == 2                  # one invented ID per episode dropped
    assert r.stats["llm"]["calls"] == len(fake.requests)           # every call logged for cost


def test_resume_skips_finished_episodes(tmp_path):
    eps = episodes()
    down_for_2 = lambda m, model: (TransientError("HTTP 503 The model is overloaded.")
                                   if m.get("task") == "moments" and m["episodes"] == [2] else None)
    fake1 = FakeModel(eps, fail=down_for_2)
    r1 = run_llm(eps, MockLLM(fake1, max_retries=2), tmp_path, series="s", episodes_per_call=1, log=quiet)
    assert not r1.complete and r1.result is None
    assert r1.pending_episodes == [2] and "503" in r1.stop_reason
    assert fake1.count("moments") == 1 + 3                          # ep 1 once, ep 2: 1 try + 2 retries
    assert not (tmp_path / "promos_ranked.json").exists()          # nothing written from a half run

    fake2 = FakeModel(eps)
    r2 = run_llm(eps, MockLLM(fake2), tmp_path, series="s", episodes_per_call=1, log=quiet)
    assert r2.complete and fake2.episodes_sent() == [2]            # episode 1 was not sent again

    fake3 = FakeModel(eps)
    r3 = run_llm(eps, MockLLM(fake3), tmp_path, series="s", episodes_per_call=1, log=quiet)
    assert r3.complete and fake3.requests == []                    # all done: no API call at all
    assert [p.id for p in r3.result.ranked] == [p.id for p in r2.result.ranked]


def test_missing_judgment_is_never_filled_in(tmp_path):
    eps = episodes()
    probe = run_llm(eps, MockLLM(FakeModel(eps)), tmp_path / "probe", series="s", log=quiet)
    victim = probe.result.ranked[0].id
    fake = FakeModel(eps, drop_judgment={victim})
    r = run_llm(eps, MockLLM(fake), tmp_path / "run", series="s", log=quiet)
    assert not r.complete and r.result is None and r.pending_clips == [victim]
    assert fake.count("judge") == 2                                # retried once in a smaller batch
    rp = run_llm(eps, MockLLM(FakeModel(eps, drop_judgment={victim})), tmp_path / "run", series="s",
                 allow_partial=True, log=quiet)
    assert rp.result is not None and rp.stats["complete"] is False
    assert victim not in [p.id for p in rp.result.ranked]
    assert next(p for p in rp.result.candidates if p.id == victim).status == "not_judged"


def test_judgment_validation_rejects_bad_scores():
    ok = {"hook_score": 4, "cliffhanger_score": "5", "standalone_score": 3.0, "hook": "h", "cliffhanger": "c"}
    assert _validate_judgment(ok)["cliffhanger_score"] == 5
    assert _validate_judgment({**ok, "hook_score": 7}) is None
    assert _validate_judgment({**ok, "standalone_score": "great"}) is None
    assert _validate_judgment({**ok, "hook": ""}) is None
    assert _validate_judgment({k: v for k, v in ok.items() if k != "cliffhanger_score"}) is None


def test_failed_test_request_sends_no_episode(tmp_path):
    eps = episodes()
    fake = FakeModel(eps, fail=lambda m, model: TransientError("HTTP 503") if m.get("task") == "preflight" else None)
    with pytest.raises(ServiceUnavailable):
        run_llm(eps, MockLLM(fake, max_retries=2), tmp_path, series="s", log=quiet)
    assert fake.count("moments") == 0 and fake.count("judge") == 0
    assert not (tmp_path / "state" / "manifest.json").exists()


def test_fallback_model_is_tested_then_pinned(tmp_path):
    eps = episodes()
    fake = FakeModel(eps, fail=lambda m, model: TransientError("HTTP 503") if model == "big" else None)
    llm = MockLLM(fake, model="big", models=["big", "small"], max_retries=1)
    r = run_llm(eps, llm, tmp_path, series="s", fallback_models=["not-listed", "small"], log=quiet)
    assert r.complete and r.stats["llm"]["model"] == "small"
    assert json.loads((tmp_path / "state" / "manifest.json").read_text())["model"] == "small"
    # rerunning the same command resumes on the pinned model
    r2 = run_llm(eps, MockLLM(FakeModel(eps), model="big", models=["big", "small"]), tmp_path, series="s",
                 fallback_models=["small"], log=quiet)
    assert r2.complete and r2.stats["llm"]["model"] == "small"
    # a different model on the same folder is refused, not mixed in
    with pytest.raises(FatalError) as e:
        run_llm(eps, MockLLM(FakeModel(eps), model="other"), tmp_path, series="s", log=quiet)
    assert e.value.kind == "state"


def test_concurrency_is_bounded(tmp_path):
    eps = episodes()
    for limit in (1, 2):
        fake = FakeModel(eps, delay=0.03)
        run_llm(eps, MockLLM(fake), tmp_path / str(limit), series="s", episodes_per_call=1, judge_batch=1,
                concurrency=limit, log=quiet)
        assert fake.max_in_flight == limit


# ------------------------------------------------------------------------- HTTP layer
class _Resp:
    def __init__(self, status, payload, headers=None):
        self.status_code, self._p, self.headers = status, payload, headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._p


OK = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}, "finishReason": "STOP"}],
      "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3, "thoughtsTokenCount": 5}}
OVERLOADED = {"error": {"code": 503, "message": "The model is overloaded. Please try again later.",
                        "status": "UNAVAILABLE"}}


def _gemini(monkeypatch, responses, **kw):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        return responses[min(len(calls), len(responses)) - 1]

    monkeypatch.setattr(L.requests, "post", fake_post)
    kw.setdefault("rpm", 0)
    kw.setdefault("backoff_base", 0.0)
    return GeminiLLM("AQ.test", "models/gemini-3.8-flash", log=quiet, **kw), calls


def test_gemini_endpoint_and_503_retry(monkeypatch):
    g, calls = _gemini(monkeypatch, [_Resp(503, OVERLOADED), _Resp(503, OVERLOADED), _Resp(200, OK)])
    rep = g.generate_json("s", "u")
    assert rep.data == {"ok": True} and rep.retries == 2 and len(calls) == 3
    url, body, headers = calls[-1]
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    assert headers == {"x-goog-api-key": "AQ.test"}
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert "temperature" not in gen and "thinkingConfig" not in gen    # Gemini 3+: leave unset
    assert rep.input_tokens == 11 and rep.output_tokens == 8


def test_gemini_gives_up_after_max_retries(monkeypatch):
    g, calls = _gemini(monkeypatch, [_Resp(503, OVERLOADED)], max_retries=3)
    with pytest.raises(ServiceUnavailable):
        g.generate_json("s", "u")
    assert len(calls) == 4


def test_backoff_is_exponential_capped_and_jittered(monkeypatch):
    g = GeminiLLM("k", "m", backoff_base=5, backoff_cap=120, log=quiet)
    monkeypatch.setattr(L.random, "uniform", lambda a, b: b)
    assert [g.backoff_delay(i) for i in range(7)] == [5, 10, 20, 40, 80, 120, 120]
    monkeypatch.setattr(L.random, "uniform", lambda a, b: a)
    assert g.backoff_delay(2) == 10                         # jitter floor is half the ceiling
    assert g.backoff_delay(0, retry_after=30) == 30         # server hint wins when longer


def test_rate_limit_429_is_retried_but_daily_quota_is_not(monkeypatch):
    per_min = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Rate limit",
                         "details": [{"retryDelay": "0s"}]}}
    g, calls = _gemini(monkeypatch, [_Resp(429, per_min), _Resp(200, OK)])
    assert g.generate_json("s", "u").data == {"ok": True} and len(calls) == 2

    daily = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                       "message": "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0",
                       "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
    g, calls = _gemini(monkeypatch, [_Resp(429, daily)])
    with pytest.raises(FatalError) as e:
        g.generate_json("s", "u")
    assert e.value.kind == "quota" and len(calls) == 1
    assert L._retry_after(_Resp(429, {"error": {"details": [{"retryDelay": "17s"}]}})) == 17.0


def test_bad_key_and_unknown_model_are_fatal(monkeypatch):
    g, _ = _gemini(monkeypatch, [_Resp(404, {"error": {"message": "models/x is not found"}})])
    with pytest.raises(FatalError) as e:
        g.generate_json("s", "u")
    assert e.value.kind == "model"
    g, _ = _gemini(monkeypatch, [_Resp(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})])
    with pytest.raises(FatalError) as e:
        g.generate_json("s", "u")
    assert e.value.kind == "auth"


def test_cut_off_or_blocked_output_is_not_used(monkeypatch):
    cut = {"candidates": [{"content": {"parts": [{"text": '{"episodes": [{"epis'}]}, "finishReason": "MAX_TOKENS"}]}
    g, _ = _gemini(monkeypatch, [_Resp(200, cut)])
    with pytest.raises(UnitError):
        g.generate_json("s", "u")
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}}
    g, _ = _gemini(monkeypatch, [_Resp(200, blocked)])
    with pytest.raises(UnitError):
        g.generate_json("s", "u")


def test_invalid_json_is_repaired_once_then_rejected(monkeypatch):
    bad = {"candidates": [{"content": {"parts": [{"text": "Sure! here you go"}]}, "finishReason": "STOP"}]}
    g, calls = _gemini(monkeypatch, [_Resp(200, bad), _Resp(200, OK)])
    assert g.generate_json("s", "u").data == {"ok": True} and len(calls) == 2
    g, calls = _gemini(monkeypatch, [_Resp(200, bad)])
    with pytest.raises(UnitError):
        g.generate_json("s", "u")
    assert len(calls) == 2


def test_thinking_level_is_sent_only_when_asked():
    g = GeminiLLM("k", "gemini-3.8-flash", thinking="low", log=quiet)
    assert g.request_body("s", "u")["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}


def test_openai_compat_falls_back_without_json_mode(monkeypatch):
    seen = []

    def fake_post(url, json=None, timeout=None, headers=None):
        seen.append(json)
        if "response_format" in json:
            return _Resp(400, {"error": "response_format not supported"})
        return _Resp(200, {"choices": [{"message": {"content": '{"x": 1}'}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    monkeypatch.setattr(L.requests, "post", fake_post)
    c = L.make_llm("custom", model="m", base_url="https://h/v1", api_key="k", rpm=0, log=quiet)
    assert c.generate_json("s", "u").data == {"x": 1}
    assert "response_format" in seen[0] and "response_format" not in seen[-1]
    assert "temperature" not in seen[-1]


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": [1, 2]} thanks') == {"a": [1, 2]}
