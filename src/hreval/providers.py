"""Model providers behind one interface.

Retry discipline (measurement-integrity rule 2): infrastructure failures —
HTTP 429, 5xx, network errors — are retried with backoff. A bad MODEL RESPONSE
is never retried. That line is the difference between measuring a model and
measuring our own patience.
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx

from .ratelimit import RateLimiter, backoff_delay

MISTRAL_URL = "https://api.mistral.ai/v1"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    pass


class QuotaNotActivated(ProviderError):
    """The key is valid but the workspace is not cleared for inference."""


class MistralProvider:
    name = "mistral"

    def __init__(self, api_key: str | None = None, *, rate_per_sec: float = 1.0,
                 max_retries: int = 5, timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not self.api_key:
            raise ProviderError("MISTRAL_API_KEY is not set")
        self._limiter = RateLimiter(rate_per_sec, burst=max(1, int(rate_per_sec)))
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=MISTRAL_URL,
            timeout=timeout,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        await self._limiter.acquire()
        resp = await self._client.get("/models")
        resp.raise_for_status()
        return sorted(m["id"] for m in resp.json().get("data", []))

    async def chat(self, *, model: str, messages: list[dict], tools: list[dict] | None = None,
                   temperature: float = 0.0, seed: int | None = None,
                   max_tokens: int = 1024) -> dict:
        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if seed is not None:
            body["random_seed"] = seed

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            await self._limiter.acquire()
            try:
                resp = await self._client.post("/chat/completions", json=body)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                await asyncio.sleep(backoff_delay(attempt))
                continue

            if resp.status_code == 429 and resp.headers.get("x-ratelimit-limit-req-minute") == "0":
                # Not throttling: the workspace has no inference allowance at all.
                # Retrying cannot fix this, so fail loudly and immediately.
                raise QuotaNotActivated(
                    f"Model {model!r} has a quota of 0 requests/minute on this account, "
                    "so no inference is permitted for it. This is a tier boundary, not an "
                    "inactive account: Mistral's free tier grants zero quota on PREMIER "
                    "models (mistral-medium, mistral-small, magistral) while open-weight "
                    "models (ministral-3b/8b, open-mistral-nemo, mistral-tiny) work "
                    "normally. Either pick an open-weight model or enable pay-as-you-go. "
                    "Retrying will not help."
                )
            if resp.status_code in RETRYABLE_STATUS:
                last_error = ProviderError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                retry_after = resp.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff_delay(attempt)
                await asyncio.sleep(delay)
                continue
            if resp.status_code >= 400:
                # Non-retryable: a bad request is our bug, surface it immediately.
                raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:400]}")

            return _normalise(resp.json())

        raise ProviderError(f"exhausted {self._max_retries} retries: {last_error}")


def _normalise(payload: dict) -> dict:
    choice = payload["choices"][0]["message"]
    usage = payload.get("usage", {}) or {}
    raw_calls = choice.get("tool_calls") or []
    calls = [
        {
            "id": c.get("id", ""),
            "name": c.get("function", {}).get("name", ""),
            "arguments": c.get("function", {}).get("arguments", "{}"),
        }
        for c in raw_calls
    ]
    return {
        "content": choice.get("content") or "",
        "tool_calls": calls,
        "raw_tool_calls": raw_calls,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


class ScriptedProvider:
    """Offline stand-in used by the tests, so the harness is testable without a key."""

    name = "scripted"

    def __init__(self, script: list[dict]):
        self._script = list(script)
        self.calls: list[dict] = []

    async def chat(self, *, model, messages, tools=None, temperature=0.0, seed=None, max_tokens=1024):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._script:
            return {"content": "", "tool_calls": [], "raw_tool_calls": [],
                    "prompt_tokens": 0, "completion_tokens": 0}
        step = self._script.pop(0)
        calls = [
            {"id": f"c{i}", "name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}
            for i, c in enumerate(step.get("tool_calls", []))
        ]
        return {
            "content": step.get("content", ""),
            "tool_calls": calls,
            "raw_tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls
            ],
            "prompt_tokens": step.get("prompt_tokens", 10),
            "completion_tokens": step.get("completion_tokens", 5),
        }
