"""LLM access: provider-agnostic caching, retry/backoff, JSON validation, usage stats.

Backends only do transport:
- OpenAICompatBackend: any /v1/chat/completions server (Ollama now, vLLM later).

Every call is cached on disk keyed by a hash of the full request, including the
provider, base URL and model, so results from different backends never mix.
Cache entries keep the original token counts and latency, so a cached rerun can
still report what the cold run cost in time and tokens.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import threading
import time
from collections import defaultdict
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .cache import DiskCache, stable_hash
from .config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_RETRY_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class PreflightError(RuntimeError):
    pass


class Completion(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    reasoning_chars: int = 0  # thinking content returned by the server (should be 0)


class LLMResult(Completion):
    latency_s: float
    cached: bool


class JsonMeta(BaseModel):
    repaired: bool = False
    fence_stripped: bool = False
    think_stripped: bool = False


# ---------------------------------------------------------------------------- usage


class PurposeUsage(BaseModel):
    calls: int = 0  # real requests
    cache_hits: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    request_s: float = 0.0  # sum of per-request latency (real requests)
    cached_input_tokens: int = 0
    cached_output_tokens: int = 0
    cached_request_s: float = 0.0  # original latency of requests served from cache
    max_output_tokens: int = 0
    reasoning_responses: int = 0  # responses that carried thinking content
    json_ok: int = 0  # valid on first attempt
    json_repaired: int = 0
    json_failed: int = 0
    fences_stripped: int = 0
    think_stripped: int = 0


class UsageTracker:
    """Per-purpose counters. Wall time per purpose is first-start..last-end of real requests."""

    def __init__(self):
        self.by_purpose: dict[str, PurposeUsage] = defaultdict(PurposeUsage)
        self._span: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def record(self, purpose: str, r: LLMResult, t_start: float | None = None, retries: int = 0) -> None:
        with self._lock:
            u = self.by_purpose[purpose]
            u.retries += retries
            u.max_output_tokens = max(u.max_output_tokens, r.output_tokens)
            u.reasoning_responses += r.reasoning_chars > 0
            if r.cached:
                u.cache_hits += 1
                u.cached_input_tokens += r.input_tokens
                u.cached_output_tokens += r.output_tokens
                u.cached_request_s += r.latency_s
            else:
                u.calls += 1
                u.input_tokens += r.input_tokens
                u.output_tokens += r.output_tokens
                u.request_s += r.latency_s
                if t_start is not None:
                    span = self._span.setdefault(purpose, [t_start, t_start + r.latency_s])
                    span[0] = min(span[0], t_start)
                    span[1] = max(span[1], t_start + r.latency_s)

    def record_json(self, purpose: str, meta: JsonMeta | None, failed: bool = False) -> None:
        with self._lock:
            u = self.by_purpose[purpose]
            if failed:
                u.json_failed += 1
                return
            assert meta is not None
            u.json_repaired += meta.repaired
            u.json_ok += not meta.repaired
            u.fences_stripped += meta.fence_stripped
            u.think_stripped += meta.think_stripped

    def summary(self) -> dict[str, dict[str, Any]]:
        out = {}
        for p, u in sorted(self.by_purpose.items()):
            span = self._span.get(p)
            wall = span[1] - span[0] if span else 0.0
            n = u.calls + u.cache_hits
            out[p] = {
                **u.model_dump(),
                "request_s": round(u.request_s, 1),
                "cached_request_s": round(u.cached_request_s, 1),
                "wall_s": round(wall, 1),
                # per-stream decode speed, and aggregate throughput across parallel requests
                "out_tok_per_s_stream": round(u.output_tokens / u.request_s, 1) if u.request_s else None,
                "out_tok_per_s_aggregate": round(u.output_tokens / wall, 1) if wall else None,
                "mean_output_tokens": round(
                    (u.output_tokens + u.cached_output_tokens) / n, 1) if n else None,
            }
        return out


# ---------------------------------------------------------------------------- backends


class Backend(Protocol):
    provider: str
    base_url: str
    model: str

    async def acomplete(self, req: dict[str, Any], schema: type[BaseModel] | None) -> Completion: ...
    def complete(self, req: dict[str, Any], schema: type[BaseModel] | None) -> Completion: ...
    async def preflight(self) -> str: ...


def same_model(a: str, b: str) -> bool:
    """Ollama treats "name" and "name:latest" as the same model."""
    norm = lambda m: m if ":" in m else f"{m}:latest"
    return norm(a) == norm(b)


class OpenAICompatBackend:
    provider = "openai-compat"

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout_s: float = 600.0, disable_thinking: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.disable_thinking = disable_thinking
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = httpx.Timeout(timeout_s, connect=10.0)
        self._headers = headers
        self._aclient: httpx.AsyncClient | None = None

    def _body(self, req: dict[str, Any], schema: type[BaseModel] | None) -> dict[str, Any]:
        msgs = []
        if req["system"]:
            msgs.append({"role": "system", "content": req["system"]})
        msgs.append({"role": "user", "content": req["prompt"]})
        body: dict[str, Any] = {"model": self.model, "messages": msgs, "temperature": req["temperature"]}
        if req["max_output_tokens"]:
            body["max_tokens"] = req["max_output_tokens"]
        thinking = req.get("thinking")  # per-call override; None = backend default
        if thinking is False or (thinking is None and self.disable_thinking):
            # Ollama's /v1 maps this to think=false; a top-level `think` is ignored there.
            body["reasoning_effort"] = "none"
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": True,
                                "schema": schema.model_json_schema()},
            }
        elif req.get("json_object"):
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _parse(d: dict[str, Any]) -> Completion:
        msg = d["choices"][0]["message"]
        u = d.get("usage") or {}
        return Completion(
            text=msg.get("content") or "",
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
            reasoning_chars=len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
        )

    @staticmethod
    def _raise_for_status(r: httpx.Response) -> None:
        if r.status_code >= 400:
            err = RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            err.code = r.status_code  # type: ignore[attr-defined]
            raise err

    async def acomplete(self, req, schema):
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self._timeout, headers=self._headers)
        r = await self._aclient.post(f"{self.base_url}/chat/completions", json=self._body(req, schema))
        self._raise_for_status(r)
        return self._parse(r.json())

    def complete(self, req, schema):
        with httpx.Client(timeout=self._timeout, headers=self._headers) as c:
            r = c.post(f"{self.base_url}/chat/completions", json=self._body(req, schema))
        self._raise_for_status(r)
        return self._parse(r.json())

    @property
    def native_url(self) -> str:
        """Ollama's native API root (the /v1 prefix stripped)."""
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    async def loaded_models(self) -> list[str]:
        """Models currently resident on an Ollama server (GET /api/ps)."""
        return [m["name"] for m in await self.residency()]

    async def residency(self) -> list[dict[str, Any]]:
        """GET /api/ps entries: name, size, size_vram, context_length, ..."""
        async with httpx.AsyncClient(timeout=10.0, headers=self._headers) as c:
            r = await c.get(f"{self.native_url}/api/ps")
            r.raise_for_status()
        return r.json().get("models", [])

    async def unload(self, model: str) -> None:
        """Ask Ollama to evict `model` now (keep_alive=0)."""
        async with httpx.AsyncClient(timeout=120.0, headers=self._headers) as c:
            r = await c.post(f"{self.native_url}/api/generate", json={"model": model, "keep_alive": 0})
            r.raise_for_status()

    async def preflight(self) -> str:
        url = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=10.0, headers=self._headers) as c:
                r = await c.get(url)
                r.raise_for_status()
        except Exception as exc:
            raise PreflightError(
                f"cannot reach {url} ({type(exc).__name__}: {exc}). "
                "Is the SSH tunnel up (port forward to the Ollama host)?"
            ) from exc
        ids = [m.get("id") for m in r.json().get("data", [])]
        if not any(same_model(self.model, i) for i in ids):
            raise PreflightError(f"model {self.model!r} not served at {self.base_url}; available: {ids}")
        return f"{self.model} @ {self.base_url}"


# ---------------------------------------------------------------------------- LLM


def strip_wrappers(text: str) -> tuple[str, bool, bool]:
    """Remove <think> blocks and ```json fences. Returns (text, fence_stripped, think_stripped)."""
    t2 = _THINK.sub("", text)
    think = t2 != text
    m = _FENCE.match(t2)
    if m:
        return m.group(1).strip(), True, think
    return t2.strip(), False, think


def _retry_delay(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None if not retryable."""
    code = getattr(exc, "code", None)
    transient = code in _RETRY_CODES or isinstance(
        exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
    )
    if not transient:
        return None
    m = re.search(r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", str(exc))
    base = float(m.group(1)) if m else min(60.0, 2.0 ** attempt)
    return base + random.uniform(0, 1.0)


class LLM:
    def __init__(
        self,
        backend: Backend,
        cache_dir: str = "cache",
        max_concurrency: int = 4,
        retry_budget_s: float = 900.0,
        tracker: UsageTracker | None = None,
    ):
        self.backend = backend
        self.cache = DiskCache(f"{cache_dir}/llm")
        self.tracker = tracker or UsageTracker()
        # Transient errors (connection refused/reset, timeouts, 429/5xx) are retried with
        # backoff until this much time has passed since the request's first failure.
        self.retry_budget_s = retry_budget_s
        self._sem = asyncio.Semaphore(max_concurrency)

    def _retry_after(self, exc: Exception, attempt: int, first_fail: float) -> float | None:
        delay = _retry_delay(exc, attempt)
        if delay is None or time.monotonic() + delay - first_fail > self.retry_budget_s:
            return None
        return delay

    @property
    def model(self) -> str:
        return self.backend.model

    @property
    def identity(self) -> dict[str, str]:
        """Provider/endpoint/model triple; part of every cache key."""
        b = self.backend
        return {"provider": b.provider, "base_url": b.base_url, "model": b.model}

    async def preflight(self) -> str:
        return await self.backend.preflight()

    def _request(self, prompt, system, schema, temperature, max_output_tokens,
                 thinking: bool | None = None, json_object: bool = False) -> dict[str, Any]:
        req = {
            **self.identity,
            "system": system,
            "prompt": prompt,
            "schema": schema.model_json_schema() if schema else None,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "thinking_off": getattr(self.backend, "disable_thinking", None),
        }
        # Only present when non-default, so existing cache keys stay valid.
        if thinking is not None:
            req["thinking"] = thinking
        if json_object:
            req["json_object"] = True
        return req

    def _from_cache(self, key: str) -> LLMResult | None:
        hit = self.cache.get(key)
        return LLMResult(cached=True, **hit) if hit is not None else None

    def _store(self, key: str, c: Completion, latency: float) -> LLMResult:
        r = LLMResult(**c.model_dump(), latency_s=latency, cached=False)
        self.cache.put(key, r.model_dump(exclude={"cached"}))
        return r

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        purpose: str = "misc",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        thinking: bool | None = None,
        json_object: bool = False,
    ) -> LLMResult:
        req = self._request(prompt, system, schema, temperature, max_output_tokens,
                            thinking, json_object)
        key = stable_hash(req)
        if (hit := self._from_cache(key)) is not None:
            self.tracker.record(purpose, hit)
            return hit
        attempt, first_fail = 0, None
        async with self._sem:
            while True:
                t0 = time.perf_counter()
                try:
                    c = await self.backend.acomplete(req, schema)
                    break
                except Exception as exc:
                    first_fail = first_fail or time.monotonic()
                    delay = self._retry_after(exc, attempt, first_fail)
                    if delay is None:
                        raise
                    attempt += 1
                    log.warning("LLM %s: %s (attempt %d), retrying in %.1fs",
                                purpose, str(exc)[:200], attempt, delay)
                    await asyncio.sleep(delay)
        r = self._store(key, c, time.perf_counter() - t0)
        self.tracker.record(purpose, r, t_start=t0, retries=attempt)
        return r

    def generate_sync(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        purpose: str = "misc",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        thinking: bool | None = None,
        json_object: bool = False,
    ) -> LLMResult:
        """Blocking variant for libraries (e.g. Mem0) that call the LLM synchronously."""
        req = self._request(prompt, system, schema, temperature, max_output_tokens,
                            thinking, json_object)
        key = stable_hash(req)
        if (hit := self._from_cache(key)) is not None:
            self.tracker.record(purpose, hit)
            return hit
        attempt, first_fail = 0, None
        while True:
            t0 = time.perf_counter()
            try:
                c = self.backend.complete(req, schema)
                break
            except Exception as exc:
                first_fail = first_fail or time.monotonic()
                delay = self._retry_after(exc, attempt, first_fail)
                if delay is None:
                    raise
                attempt += 1
                log.warning("LLM %s: %s (attempt %d), retrying in %.1fs",
                            purpose, str(exc)[:200], attempt, delay)
                time.sleep(delay)
        r = self._store(key, c, time.perf_counter() - t0)
        self.tracker.record(purpose, r, t_start=t0, retries=attempt)
        return r

    async def generate_json_meta(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        purpose: str = "misc",
        max_output_tokens: int | None = None,
        repair_attempts: int = 1,
    ) -> tuple[T, JsonMeta]:
        """Structured output (schema sent to the server), validated by Pydantic.
        On a validation failure the error is fed back for repair (also cached)."""
        meta = JsonMeta()
        r = await self.generate(prompt, system=system, schema=schema, purpose=purpose,
                                max_output_tokens=max_output_tokens)
        for i in range(repair_attempts + 1):
            text, fence, think = strip_wrappers(r.text)
            meta.fence_stripped |= fence
            meta.think_stripped |= think
            try:
                obj = schema.model_validate_json(text)
                self.tracker.record_json(purpose, meta)
                return obj, meta
            except ValidationError as err:
                if i == repair_attempts:
                    self.tracker.record_json(purpose, None, failed=True)
                    raise
                meta.repaired = True
                repair = (
                    f"{prompt}\n\nYour previous answer was invalid:\n{r.text[:4000]}\n\n"
                    f"Validation errors:\n{str(err)[:2000]}\n\nReturn corrected JSON only."
                )
                r = await self.generate(repair, system=system, schema=schema, purpose=purpose,
                                        max_output_tokens=max_output_tokens)
        raise AssertionError("unreachable")

    async def generate_json(self, prompt: str, schema: type[T], **kw: Any) -> T:
        return (await self.generate_json_meta(prompt, schema, **kw))[0]


# ---------------------------------------------------------------------------- factories


def make_llm(s: Settings, tracker: UsageTracker | None = None) -> LLM:
    """The extraction/answering model (local, OpenAI-compatible)."""
    backend = OpenAICompatBackend(s.llm_base_url, s.llm_model, s.llm_api_key,
                                  s.llm_timeout_s, s.llm_disable_thinking)
    return LLM(backend, s.cache_dir, s.llm_max_concurrency, s.llm_retry_budget_s, tracker)


def make_judge(s: Settings, tracker: UsageTracker | None = None) -> tuple[LLM, dict[str, str]]:
    """The local judge (GRAPHMEM_JUDGE_MODEL, default: the answer model), thinking off.
    Returns (llm, record); the record goes into every results file."""
    backend = OpenAICompatBackend(s.judge_base_url or s.llm_base_url, s.judge_model or s.llm_model,
                                  s.llm_api_key, s.llm_timeout_s, disable_thinking=True)
    llm = LLM(backend, s.cache_dir, s.llm_max_concurrency, s.llm_retry_budget_s, tracker)
    return llm, {**llm.identity, "thinking": "off"}
