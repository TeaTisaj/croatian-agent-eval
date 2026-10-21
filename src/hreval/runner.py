"""Resumable matrix executor.

Free-tier 429s will kill a long run. Every completed cell is checkpointed in
SQLite keyed on (scenario, tier, condition, repeat), so a re-run skips what is
already done. Without this you restart from zero, which is how a weekend dies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import tomllib
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .domain.store import Store
from .harness.agent import run_scenario
from .providers import MistralProvider
from .scenarios.schema import load as load_scenarios

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "results" / "results.db"
TRACE_DIR = ROOT / "traces"

# condition -> (which user-turn language, which harness language)
CONDITIONS = {"HR_HR": ("hr", "hr"), "HR_EN": ("hr", "en"), "EN_EN": ("en", "en")}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  scenario_id TEXT NOT NULL,
  tier        TEXT NOT NULL,
  model       TEXT NOT NULL,
  condition   TEXT NOT NULL,
  repeat      INTEGER NOT NULL,
  stop_reason TEXT,
  trace_path  TEXT,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  n_model_calls     INTEGER,
  latency_ms        INTEGER,
  created_at  TEXT,
  PRIMARY KEY (scenario_id, tier, condition, repeat)
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def done_cells(conn) -> set[tuple]:
    return {tuple(r) for r in conn.execute("SELECT scenario_id, tier, condition, repeat FROM runs")}


async def _one(sem, provider, conn, lock, cfg, tier, model, scenario, condition, repeat, frozen_now):
    turn_lang, harness_lang = CONDITIONS[condition]
    turns = scenario.turns_hr if turn_lang == "hr" else scenario.turns_en
    store = Store(frozen_now=frozen_now)  # one Store per run, never shared

    async with sem:
        result = await run_scenario(
            provider=provider, model=model, user_turns=turns, harness_lang=harness_lang,
            store=store, frozen_now=frozen_now,
            temperature=cfg["matrix"]["temperature"], seed=repeat,
            max_tool_calls=cfg["matrix"]["max_tool_calls"], max_turns=cfg["matrix"]["max_turns"],
        )

    name = f"{scenario.id}__{tier}__{condition}__r{repeat}.json"
    path = TRACE_DIR / name
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace = result.as_trace(
        scenario_id=scenario.id, tier=tier, model=model, condition=condition,
        repeat=repeat, harness_lang=harness_lang, turn_lang=turn_lang,
    )
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")

    async with lock:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (scenario.id, tier, model, condition, repeat, result.stop_reason, name,
             result.prompt_tokens, result.completion_tokens, result.n_model_calls,
             result.latency_ms, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
    return result.prompt_tokens + result.completion_tokens


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", help="large|medium|small; default all pinned")
    ap.add_argument("--condition", help="HR_HR|HR_EN|EN_EN; default all")
    ap.add_argument("--limit", type=int, help="cap scenarios, for a smoke run")
    ap.add_argument("--repeats", type=int, help="override configured repeats")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = tomllib.loads((ROOT / "config" / "run.toml").read_text())
    lock_file = ROOT / "config" / "models.lock.json"
    if not lock_file.exists():
        print("config/models.lock.json missing - run `make smoke` first to pin model ids")
        return 1
    tiers: dict[str, str] = json.loads(lock_file.read_text())["tiers"]
    if args.tier:
        tiers = {args.tier: tiers[args.tier]}

    conditions = [args.condition] if args.condition else cfg["matrix"]["conditions"]
    repeats = args.repeats or cfg["matrix"]["repeats"]
    scenarios = load_scenarios()
    if args.limit:
        scenarios = scenarios[: args.limit]

    frozen_now = datetime.fromisoformat(cfg["clock"]["frozen_now"])
    conn = connect()
    already = done_cells(conn)

    cells = [
        (tier, model, s, cond, r)
        for tier, model in tiers.items()
        for cond in conditions
        for s in scenarios
        for r in range(1, repeats + 1)
        if (s.id, tier, cond, r) not in already
    ]
    total = len(tiers) * len(conditions) * len(scenarios) * repeats
    skipped = total - len(cells)
    print(f"matrix {total} cells for this selection | {skipped} already done | {len(cells)} to run")
    if not cells:
        return 0

    provider = MistralProvider(
        rate_per_sec=cfg["limits"]["requests_per_second"],
        max_retries=cfg["limits"]["max_retries"],
    )
    sem = asyncio.Semaphore(cfg["limits"]["max_concurrency"])
    db_lock = asyncio.Lock()
    ceiling = cfg["limits"]["token_budget_ceiling"]
    spent = 0

    try:
        tasks = [
            asyncio.create_task(_one(sem, provider, conn, db_lock, cfg, t, m, s, c, r, frozen_now))
            for (t, m, s, c, r) in cells
        ]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            spent += await coro
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} runs | {spent:,} tokens")
            if spent > ceiling:  # budget guard: never silently burn the free quota
                print(f"ABORT: token ceiling {ceiling:,} exceeded")
                for t in tasks:
                    t.cancel()
                return 2
    finally:
        await provider.aclose()
        conn.close()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
