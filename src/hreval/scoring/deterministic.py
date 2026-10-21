"""Deterministic scoring. Reads traces, never the API.

Because scoring is decoupled from execution, a scorer bug costs a re-score and
not 972 model calls. Per-field argument accuracy is the PRIMARY metric: at
n=36 scenarios, per-scenario completion is underpowered, while field-level
comparison yields several hundred paired observations.
"""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path

from ..scenarios.schema import Scenario, load as load_scenarios

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "results" / "results.db"
TRACE_DIR = ROOT / "traces"

SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
  scenario_id TEXT, tier TEXT, condition TEXT, repeat INTEGER,
  bucket TEXT, axes TEXT,
  completed INTEGER, tools_ok INTEGER,
  args_correct INTEGER, args_total INTEGER,
  step_efficiency REAL, lang_drift INTEGER,
  failure_cause TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER,
  PRIMARY KEY (scenario_id, tier, condition, repeat)
);
"""


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder
    # Binary Croatian-family vs English only. lingua cannot reliably separate
    # HR from SR, and is deliberately not asked to.
    return LanguageDetectorBuilder.from_languages(Language.CROATIAN, Language.ENGLISH).build()


def detect_drift(texts: list[str], turn_lang: str) -> bool:
    if turn_lang != "hr":
        return False
    joined = " ".join(t for t in texts if t and len(t.strip()) > 15)
    if not joined:
        return False
    from lingua import Language
    return _detector().detect_language_of(joined) == Language.ENGLISH


def _confirmed(trace: dict) -> dict | None:
    state = trace.get("terminal_state") or {}
    rows = state.get("confirmed") or []
    return rows[0] if len(rows) == 1 else None


def score_tools(trace: dict, s: Scenario) -> tuple[bool, list[str]]:
    names = [c["name"] for c in trace["tool_calls"]]
    problems = []
    for tool in s.expected.required_tools:
        if tool not in names:
            problems.append(f"missing:{tool}")
    for tool in s.expected.forbidden_tools:
        if tool in names:
            problems.append(f"forbidden:{tool}")
    for before, after in s.expected.order_constraints:
        if before in names and after in names and names.index(before) > names.index(after):
            problems.append(f"order:{before}<{after}")
    return not problems, problems


def score_outcome(trace: dict, s: Scenario) -> bool:
    state = trace.get("terminal_state") or {}
    n = state.get("n_confirmed", 0)
    names = [c["name"] for c in trace["tool_calls"]]
    if s.expected.outcome == "no_booking":
        return n == 0 and "confirm_booking" not in names and "create_booking_hold" not in names
    if s.expected.outcome == "cancelled":
        return n == 0 and "cancel_booking" in names
    booking = _confirmed(trace)
    if booking is None:
        return False
    return all(str(booking.get(k)) == str(v) for k, v in s.expected.final_args.items())


def score_args(trace: dict, s: Scenario) -> tuple[int, int]:
    """Per-field. This is where the statistical power lives."""
    if not s.expected.final_args:
        return 0, 0
    booking = _confirmed(trace)
    total = len(s.expected.final_args)
    if booking is None:
        return 0, total
    return sum(str(booking.get(k)) == str(v) for k, v in s.expected.final_args.items()), total


def classify_failure(trace: dict, s: Scenario, completed: bool, drift: bool) -> str:
    if completed:
        return "none"
    if trace["stop_reason"] == "loop_exhausted":
        return "loop_exhausted"
    if trace["stop_reason"] == "provider_error":
        return "provider_error"
    calls = trace["tool_calls"]
    if any(c.get("malformed") for c in calls):
        return "malformed_tool_call"
    if any(c.get("unknown_tool") for c in calls):
        return "hallucinated_tool"
    if not calls:
        return "no_tool_call"
    names = [c["name"] for c in calls]
    # Finalising a booking that should never have existed is the most severe
    # violation, so it wins over any other forbidden-tool use.
    if "confirm_booking" in s.expected.forbidden_tools and "confirm_booking" in names:
        return "premature_finalize"
    if any(tool in names for tool in s.expected.forbidden_tools):
        return "out_of_scope_action"
    booking = _confirmed(trace)
    if booking and s.expected.final_args:
        if str(booking.get("start")) != str(s.expected.final_args.get("start")):
            return "bad_arg_date"
        for field in ("staff_id", "service_id", "location_id"):
            if field in s.expected.final_args and str(booking.get(field)) != str(s.expected.final_args[field]):
                return "bad_arg_entity"
    if "confirm_booking" in s.expected.required_tools and "confirm_booking" not in names:
        return "never_finalised"
    if drift:
        return "lang_drift"
    return "other"


def main() -> int:
    scenarios = {s.id: s for s in load_scenarios()}
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    rows = conn.execute("SELECT scenario_id, tier, condition, repeat, trace_path FROM runs").fetchall()
    if not rows:
        print("no runs in results.db yet")
        return 1

    n = 0
    for scenario_id, tier, condition, repeat, trace_path in rows:
        path = TRACE_DIR / trace_path
        if not path.exists():
            continue
        trace = json.loads(path.read_text(encoding="utf-8"))
        s = scenarios[scenario_id]

        drift = detect_drift(trace.get("assistant_texts", []), trace.get("turn_lang", "hr"))
        completed = score_outcome(trace, s)
        tools_ok, _ = score_tools(trace, s)
        correct, total = score_args(trace, s)
        calls = trace["usage"]["n_model_calls"] or 1
        conn.execute(
            "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (scenario_id, tier, condition, repeat, s.bucket, ",".join(s.axes),
             int(completed), int(tools_ok), correct, total,
             calls / s.min_turns, int(drift),
             classify_failure(trace, s, completed, drift),
             trace["usage"]["prompt_tokens"], trace["usage"]["completion_tokens"]),
        )
        n += 1
    conn.commit()

    print(f"scored {n} runs\n")
    for tier, cond, c, t, a, at in conn.execute(
        "SELECT tier, condition, AVG(completed), COUNT(*), SUM(args_correct), SUM(args_total) "
        "FROM scores GROUP BY tier, condition ORDER BY tier, condition"
    ):
        field_acc = (a / at) if at else 0.0
        print(f"  {tier:7s} {cond:6s}  completion {c:5.1%} (n={t:3d})  field-accuracy {field_acc:5.1%}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
