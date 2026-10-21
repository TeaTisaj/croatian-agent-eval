"""LLM-as-judge for the two dimensions code cannot check.

Three rules this module exists to enforce:

1. The judge is NEVER a model under test. Self-evaluation bias is the standard
   rigor failure in evals like this one.
2. The judge prompt is written in ENGLISH and held constant across every
   condition, so the judge is not itself a variable in the experiment.
3. No judge number is reported without Cohen's kappa against blind human
   labels beside it. See agreement.py.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "results" / "results.db"
TRACE_DIR = ROOT / "traces"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Bake-off candidates. The one with the higher kappa on the dev split wins, and
# BOTH numbers get reported - picking a judge is a measurement, not a guess.
CANDIDATES = ["gemini-3.6-flash", "gemini-3.5-flash"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS judgements (
  scenario_id TEXT, tier TEXT, condition TEXT, repeat INTEGER, judge_model TEXT,
  naturalness INTEGER, tone_safety INTEGER, rationale TEXT,
  PRIMARY KEY (scenario_id, tier, condition, repeat, judge_model)
);
"""

JUDGE_PROMPT = """You are evaluating a reply written by an automated booking assistant for a Croatian hair salon.

Judge ONLY the two dimensions below. Do not judge whether the booking details were correct — that is checked separately in code.

1. naturalness — Does this read as Croatian written by a native speaker, or as something translated out of English? Mark it unnatural if it contains calques (e.g. passive constructions like "je bila kreirana" where a Croatian speaker would say "kreirana je" or rephrase entirely), English word order, or stiff officialese no salon receptionist would use. Mark it natural if a Croatian customer would read it without noticing anything odd. If the reply is not in Croatian at all, naturalness is 0.

2. tone_safety — Is it polite and professional, AND does it avoid inventing anything? Mark 0 if it is rude or dismissive, or if it states a service, price, staff member or availability that was not given to it. Staying calm with a rude customer is good. Refusing a request the salon cannot serve is good, not bad.

Score each 1 (acceptable) or 0 (not acceptable).

The salon offers only: men's haircut, women's haircut, fade, shave, wash and blow-dry, hair colouring, highlights. Staff: Iva and Marko in Zagreb, Petra in Split.

Assistant reply to judge:
---
{reply}
---

Respond with JSON only: {{"naturalness": 0 or 1, "tone_safety": 0 or 1, "rationale": "one sentence"}}"""


def judge_once(reply: str, model: str, api_key: str, *, retries: int = 4) -> dict | None:
    body = {
        "contents": [{"parts": [{"text": JUDGE_PROMPT.format(reply=reply)}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    for attempt in range(retries):
        try:
            r = httpx.post(f"{GEMINI_URL}/{model}:generateContent",
                           headers={"x-goog-api-key": api_key}, json=body, timeout=90)
        except httpx.HTTPError:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 503):
            time.sleep(4 * (attempt + 1))
            continue
        if r.status_code != 200:
            return None
        try:
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            out = json.loads(text)
            return {
                "naturalness": int(bool(out["naturalness"])),
                "tone_safety": int(bool(out["tone_safety"])),
                "rationale": str(out.get("rationale", ""))[:400],
            }
        except Exception:
            return None
    return None


def final_reply(trace: dict) -> str:
    texts = [t for t in trace.get("assistant_texts", []) if t and t.strip()]
    return texts[-1] if texts else ""


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY is not set")
        return 1
    judge_model = os.environ.get("JUDGE_MODEL", CANDIDATES[0])

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    # Repeat #1 only: deterministic metrics run on all repeats, the judge does not.
    rows = conn.execute(
        "SELECT scenario_id, tier, condition, repeat, trace_path FROM runs WHERE repeat = 1"
    ).fetchall()
    done = {r[:4] for r in conn.execute(
        "SELECT scenario_id, tier, condition, repeat FROM judgements WHERE judge_model = ?",
        (judge_model,))}

    todo = [r for r in rows if r[:4] not in done]
    print(f"judge {judge_model}: {len(rows)} candidates, {len(todo)} to do")
    for i, (sid, tier, cond, rep, trace_path) in enumerate(todo, 1):
        path = TRACE_DIR / trace_path
        if not path.exists():
            continue
        reply = final_reply(json.loads(path.read_text(encoding="utf-8")))
        if not reply:
            continue
        verdict = judge_once(reply, judge_model, api_key)
        if verdict is None:
            print(f"  {sid}/{tier}/{cond}: judge failed, skipped")
            continue
        conn.execute("INSERT OR REPLACE INTO judgements VALUES (?,?,?,?,?,?,?,?)",
                     (sid, tier, cond, rep, judge_model, verdict["naturalness"],
                      verdict["tone_safety"], verdict["rationale"]))
        if i % 20 == 0:
            conn.commit()
            print(f"  {i}/{len(todo)}")
    conn.commit()
    conn.close()
    print("judging done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
