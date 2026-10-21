"""Hand-rolled agent loop.

Deliberately NOT LangGraph/LangChain (README §Measurement integrity, rule 4):
those frameworks inject their own English ReAct scaffolding into the context,
which in a language-controlled experiment is uncontrolled English contaminating
the HR_HR condition. Every token in the context below is one we chose.

Also, per rule 1: malformed tool calls are RECORDED as failures, never repaired
or re-prompted. Hiding model weakness is the one thing this harness must not do.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..domain.store import Store
from ..domain.tools import TOOL_NAMES, dispatch, tool_specs
from .prompts import system_prompt

TRACE_SCHEMA_VERSION = 1


@dataclass
class Step:
    kind: str  # "user" | "assistant" | "tool_call" | "tool_result" | "error"
    payload: Any
    latency_ms: int | None = None


@dataclass
class RunResult:
    steps: list[Step] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    assistant_texts: list[str] = field(default_factory=list)
    terminal_state: dict | None = None
    stop_reason: str = "completed"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    n_model_calls: int = 0

    def as_trace(self, **meta) -> dict:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            **meta,
            "stop_reason": self.stop_reason,
            "tool_calls": self.tool_calls,
            "assistant_texts": self.assistant_texts,
            "terminal_state": self.terminal_state,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "n_model_calls": self.n_model_calls,
                "latency_ms": self.latency_ms,
            },
            "steps": [{"kind": s.kind, "payload": s.payload, "latency_ms": s.latency_ms} for s in self.steps],
        }


async def run_scenario(
    *,
    provider,
    model: str,
    user_turns: list[str],
    harness_lang: str,
    store: Store,
    frozen_now,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tool_calls: int = 8,
    max_turns: int = 10,
) -> RunResult:
    """Drive one scenario to completion.

    user_turns are delivered one at a time: the next is only sent after the
    model produces a text reply, which is what makes multi-turn state a real
    test rather than a transcript replay.
    """
    result = RunResult()
    specs = tool_specs(harness_lang)
    messages: list[dict] = [{"role": "system", "content": system_prompt(harness_lang, frozen_now)}]

    pending = list(user_turns)
    tool_calls_made = 0
    turns = 0
    started = time.monotonic()
    # A user turn is delivered only when the conversation is actually waiting
    # on the customer: at the start, and after each assistant TEXT reply.
    # Delivering one straight after a tool result would put words in the
    # customer's mouth before the assistant had said anything.
    awaiting_user = True

    while turns < max_turns:
        turns += 1
        if awaiting_user and pending:
            nxt = pending.pop(0)
            messages.append({"role": "user", "content": nxt})
            result.steps.append(Step("user", nxt))
            awaiting_user = False

        call_start = time.monotonic()
        try:
            reply = await provider.chat(
                model=model, messages=messages, tools=specs,
                temperature=temperature, seed=seed,
            )
        except Exception as exc:  # infrastructure failure, not a model failure
            result.stop_reason = "provider_error"
            result.steps.append(Step("error", f"{type(exc).__name__}: {exc}"))
            break
        latency = int((time.monotonic() - call_start) * 1000)
        result.n_model_calls += 1
        result.prompt_tokens += reply.get("prompt_tokens", 0)
        result.completion_tokens += reply.get("completion_tokens", 0)

        content = reply.get("content") or ""
        raw_calls = reply.get("tool_calls") or []

        if raw_calls:
            messages.append({"role": "assistant", "content": content, "tool_calls": reply["raw_tool_calls"]})
            for call in raw_calls:
                if tool_calls_made >= max_tool_calls:
                    result.stop_reason = "loop_exhausted"
                    break
                tool_calls_made += 1
                name = call["name"]
                # Rule 1: parse failures are recorded, never repaired.
                try:
                    args = json.loads(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
                    if not isinstance(args, dict):
                        raise ValueError("arguments is not an object")
                    malformed = False
                except Exception:
                    args, malformed = {}, True

                record = {"name": name, "arguments": args, "malformed": malformed,
                          "unknown_tool": name not in TOOL_NAMES}
                result.tool_calls.append(record)
                result.steps.append(Step("tool_call", record, latency))

                if malformed:
                    observation = {"error": "malformed_tool_call",
                                   "message": "Arguments were not valid JSON object."}
                elif name not in TOOL_NAMES:
                    observation = {"error": "unknown_tool", "message": f"No tool named {name}"}
                else:
                    observation = dispatch(store, name, args, harness_lang)

                result.steps.append(Step("tool_result", observation))
                messages.append({
                    "role": "tool", "name": name,
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(observation, ensure_ascii=False),
                })
            if result.stop_reason == "loop_exhausted":
                break
            continue

        # Plain text reply: the assistant has handed the conversation back.
        messages.append({"role": "assistant", "content": content})
        result.assistant_texts.append(content)
        result.steps.append(Step("assistant", content, latency))
        if not pending:
            break
        awaiting_user = True

    else:
        result.stop_reason = "max_turns"

    result.latency_ms = int((time.monotonic() - started) * 1000)
    result.terminal_state = store.terminal_state()
    return result
