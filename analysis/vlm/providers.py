"""
Requests to the three vision-language models (ANALYSIS_PLAN.md, section 7.3): one image, one new request, one structured
answer. No conversation history, no tools, no web search, no file name; sampling and reasoning settings are the defaults
of each model (nothing is set). Only the standard library is used. Keys are read from config.json, never printed.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = ["spatial_geometric_coherence", "proportional_coherence", "apparent_constructability", "component_coherence",
          "lighting", "material_rendering", "photographic_realism", "photographic_composition", "overall_visual_quality"]
NOTES = ["architectural_note", "representation_note"]
PROVIDERS = ["openai", "anthropic", "google"]
NO_CREDIT = ("insufficient_quota", "credit balance", "billing", "exceeded your current quota")


def load_config() -> dict:
    with open(os.path.join(HERE, "config.json"), encoding="utf-8-sig") as f:
        return json.load(f)


def path(cfg: dict, name: str) -> str:
    p = cfg["paths"][name]
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def schema(notes: bool, bounded: bool) -> dict:
    """JSON schema of the answer. `bounded`: integer with minimum/maximum (Google); otherwise an integer enum."""
    score = {"type": "integer", "minimum": 1, "maximum": 7} if bounded else {"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7]}
    props = {k: dict(score) for k in SCORES}
    if notes:
        props.update({k: {"type": "string"} for k in NOTES})
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _http(url: str, headers: dict, body: dict | None, timeout: int = 300):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode("utf-8"),
                                 headers={"content-type": "application/json", **headers}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(text)
        except ValueError:
            return e.code, {"error": text[:2000]}
    except Exception as e:  # network error, timeout
        return 0, {"error": f"{type(e).__name__}: {e}"}


def list_models(provider: str, cfg: dict) -> tuple[int, list[str]]:
    key = cfg["providers"][provider]["api_key"]
    if provider == "openai":
        s, d = _http("https://api.openai.com/v1/models", {"authorization": f"Bearer {key}"}, None)
        return s, [m["id"] for m in d.get("data", [])] if s == 200 else [json.dumps(d)[:300]]
    if provider == "anthropic":
        s, d = _http("https://api.anthropic.com/v1/models?limit=1000", {"x-api-key": key, "anthropic-version": "2023-06-01"}, None)
        return s, [m["id"] for m in d.get("data", [])] if s == 200 else [json.dumps(d)[:300]]
    s, d = _http("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000", {"x-goog-api-key": key}, None)
    return s, [m["name"].split("/")[-1] for m in d.get("models", [])] if s == 200 else [json.dumps(d)[:300]]


def request_body(provider: str, model: str, image: bytes, prompt: str, notes: bool, variant: dict | None = None) -> tuple[str, dict]:
    """URL and body of one evaluation request. Image first, then the rubric text, in one user message."""
    b64 = base64.b64encode(image).decode("ascii")
    variant = variant or {}
    if provider == "openai":
        return "https://api.openai.com/v1/responses", {
            "model": model, "store": False,
            "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": variant.get("detail", "original")},
                {"type": "input_text", "text": prompt}]}],
            "text": {"format": {"type": "json_schema", "name": "evaluation", "strict": True, "schema": schema(notes, False)}}}
    if provider == "anthropic":
        return "https://api.anthropic.com/v1/messages", {
            "model": model, "max_tokens": variant.get("max_tokens", 4096),
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt}]}],
            "output_config": {"format": {"type": "json_schema", "schema": schema(notes, False)}}}
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", {
        "contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": b64}}, {"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": schema(notes, True)}}


def headers(provider: str, key: str) -> dict:
    if provider == "openai":
        return {"authorization": f"Bearer {key}"}
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"x-goog-api-key": key}


def extract(provider: str, d: dict) -> tuple[str | None, str | None, dict, str | None]:
    """(answer text, model identifier returned, token usage, stop/refusal information) from a raw response."""
    try:
        if provider == "openai":
            text, refusal = None, None
            for item in d.get("output", []):
                for c in item.get("content", []) if item.get("type") == "message" else []:
                    if c.get("type") == "output_text":
                        text = (text or "") + c.get("text", "")
                    elif c.get("type") == "refusal":
                        refusal = c.get("refusal")
            return text, d.get("model"), d.get("usage") or {}, refusal or d.get("status")
        if provider == "anthropic":
            text = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text") or None
            return text, d.get("model"), d.get("usage") or {}, d.get("stop_reason")
        cand = (d.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought")) or None
        stop = cand.get("finishReason") or (d.get("promptFeedback") or {}).get("blockReason")
        return text, d.get("modelVersion"), d.get("usageMetadata") or {}, stop
    except Exception as e:
        return None, None, {}, f"unreadable response: {e}"


def validate(text: str | None, notes: bool) -> dict | None:
    """The answer as a dictionary if it has exactly the required fields with integer scores from 1 to 7, else None."""
    if not text:
        return None
    try:
        o = json.loads(text)
    except ValueError:
        return None
    want = SCORES + (NOTES if notes else [])
    if not isinstance(o, dict) or set(o) != set(want):
        return None
    if any(type(o[k]) is not int or not 1 <= o[k] <= 7 for k in SCORES):
        return None
    if notes and any(not isinstance(o[k], str) for k in NOTES):
        return None
    return o


def evaluate(provider: str, cfg: dict, image: bytes, prompt: str, notes: bool, variant: dict | None = None) -> dict:
    """One request (network errors and rate limits are retried with a pause: they are not answers of the model)."""
    p = cfg["providers"][provider]
    url, body = request_body(provider, p["model"], image, prompt, notes, variant)
    status, d, t0 = 0, {}, time.time()
    for attempt in range(6):
        t0 = time.time()
        status, d = _http(url, headers(provider, p["api_key"]), body)
        if status not in (0, 408, 409, 429, 500, 502, 503, 504, 529):
            break
        if any(w in json.dumps(d).lower() for w in NO_CREDIT):      # no credit left: waiting does not help
            return dict(http_status=status, duration_s=round(time.time() - t0, 2), model_returned=None, usage={}, stop=None, answer=None, outcome="no_credit", raw=d)
        time.sleep(min(90, 5 * 2 ** attempt))
    text, model, usage, stop = extract(provider, d) if status == 200 else (None, None, {}, None)
    answer = validate(text, notes)
    if status != 200 and any(w in json.dumps(d).lower() for w in NO_CREDIT):
        return dict(http_status=status, duration_s=round(time.time() - t0, 2), model_returned=None, usage={}, stop=None, answer=None, outcome="no_credit", raw=d)
    return dict(http_status=status, duration_s=round(time.time() - t0, 2), model_returned=model, usage=usage, stop=stop,
                answer=answer, outcome="ok" if answer else ("http_error" if status != 200 else "invalid_or_refused"), raw=d)
