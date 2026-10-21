"""Human validation of the judge: blind labelling + Cohen's kappa.

The split matters. Items are assigned deterministically to dev (20) or test
(40). The judge prompt may be iterated against DEV ONLY; the reported kappa is
computed on TEST, which is never looked at while tuning. Reporting kappa on the
same items you tuned against is judge-prompt overfitting, and it is the most
common flaw in evaluations of this shape.

Labels must come from a native Croatian speaker. The CLI shows the reply and
nothing else - no judge verdict, no model name, no condition - so the label
cannot be anchored by what the judge said.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "results" / "results.db"
TRACE_DIR = ROOT / "traces"

N_SAMPLE = 60
N_DEV = 20

SCHEMA = """
CREATE TABLE IF NOT EXISTS human_labels (
  scenario_id TEXT, tier TEXT, condition TEXT, repeat INTEGER,
  split TEXT, naturalness INTEGER, tone_safety INTEGER,
  PRIMARY KEY (scenario_id, tier, condition, repeat)
);
"""


def _split_for(key: str) -> str:
    """Deterministic, so the dev/test assignment cannot drift between runs."""
    digest = hashlib.sha256(key.encode()).hexdigest()
    return "dev" if int(digest[:8], 16) % 3 == 0 else "test"


def cohens_kappa(a: list[int], b: list[int]) -> float | None:
    """Two raters, binary categories."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in {0, 1})
    if pe == 1.0:
        return None  # no variance; kappa undefined
    return (po - pe) / (1 - pe)


def interpret(k: float) -> str:
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate - BELOW THRESHOLD, demote judge metrics"
    if k < 0.80: return "substantial"
    return "almost perfect"


def cmd_sample(conn) -> int:
    """Stratified sample of Croatian-input runs, balanced across condition and bucket."""
    rows = conn.execute(
        "SELECT r.scenario_id, r.tier, r.condition, r.repeat, r.trace_path, s.bucket "
        "FROM runs r JOIN scores s USING (scenario_id, tier, condition, repeat) "
        "WHERE r.repeat = 1 AND r.condition IN ('HR_HR','HR_EN')"
    ).fetchall()
    if not rows:
        print("no scored HR runs yet - run the matrix and `make score` first")
        return 1
    buckets: dict[tuple, list] = {}
    for row in rows:
        buckets.setdefault((row[2], row[5]), []).append(row)
    rng = random.Random(0)
    picked, pool = [], [list(v) for v in buckets.values()]
    for group in pool:
        rng.shuffle(group)
    while len(picked) < min(N_SAMPLE, len(rows)):
        progressed = False
        for group in pool:
            if group and len(picked) < N_SAMPLE:
                picked.append(group.pop())
                progressed = True
        if not progressed:
            break
    for sid, tier, cond, rep, _tp, _b in picked:
        key = f"{sid}|{tier}|{cond}|{rep}"
        conn.execute(
            "INSERT OR IGNORE INTO human_labels (scenario_id,tier,condition,repeat,split,naturalness,tone_safety)"
            " VALUES (?,?,?,?,?,NULL,NULL)", (sid, tier, cond, rep, _split_for(key)))
    conn.commit()
    counts = Counter(r[0] for r in conn.execute("SELECT split FROM human_labels"))
    print(f"sampled {len(picked)} items | dev {counts['dev']} / test {counts['test']}")
    print("now run:  make label")
    return 0


def cmd_label(conn) -> int:
    todo = conn.execute(
        "SELECT h.scenario_id, h.tier, h.condition, h.repeat, r.trace_path "
        "FROM human_labels h JOIN runs r USING (scenario_id, tier, condition, repeat) "
        "WHERE h.naturalness IS NULL"
    ).fetchall()
    if not todo:
        print("nothing left to label")
        return 0
    print(f"{len(todo)} items to label. Blind: you see only the reply.\n"
          "For each, answer 1 (acceptable) or 0. Ctrl-C to stop; progress is saved.\n")
    for i, (sid, tier, cond, rep, tp) in enumerate(todo, 1):
        path = TRACE_DIR / tp
        if not path.exists():
            continue
        trace = json.loads(path.read_text(encoding="utf-8"))
        texts = [t for t in trace.get("assistant_texts", []) if t.strip()]
        if not texts:
            continue
        print("=" * 70)
        print(f"[{i}/{len(todo)}]\n\n{texts[-1]}\n")
        try:
            nat = input("  natural Croatian, not translated?  (1/0) ").strip()
            tone = input("  polite AND invents nothing?        (1/0) ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nstopped; progress saved")
            break
        if nat not in ("0", "1") or tone not in ("0", "1"):
            print("  skipped (answer 1 or 0)")
            continue
        conn.execute("UPDATE human_labels SET naturalness=?, tone_safety=? "
                     "WHERE scenario_id=? AND tier=? AND condition=? AND repeat=?",
                     (int(nat), int(tone), sid, tier, cond, rep))
        conn.commit()
    return 0


def cmd_kappa(conn) -> int:
    models = [r[0] for r in conn.execute("SELECT DISTINCT judge_model FROM judgements")]
    if not models:
        print("no judgements yet - run `make judge` first")
        return 1
    print(f"{'judge':22s} {'split':6s} {'n':>4s}  {'naturalness':>12s}  {'tone_safety':>12s}")
    print("-" * 66)
    best = None
    for model in models:
        for split in ("dev", "test"):
            rows = conn.execute(
                "SELECT h.naturalness, h.tone_safety, j.naturalness, j.tone_safety "
                "FROM human_labels h JOIN judgements j USING (scenario_id, tier, condition, repeat) "
                "WHERE j.judge_model = ? AND h.split = ? AND h.naturalness IS NOT NULL",
                (model, split)).fetchall()
            if not rows:
                continue
            k_nat = cohens_kappa([r[0] for r in rows], [r[2] for r in rows])
            k_tone = cohens_kappa([r[1] for r in rows], [r[3] for r in rows])
            fmt = lambda k: f"{k:12.3f}" if k is not None else f"{'n/a':>12s}"
            print(f"{model:22s} {split:6s} {len(rows):4d}  {fmt(k_nat)}  {fmt(k_tone)}")
            if split == "test" and k_nat is not None and (best is None or k_nat > best[1]):
                best = (model, k_nat)
    if best:
        print(f"\nselected judge: {best[0]}  (naturalness kappa {best[1]:.3f} on held-out test)")
        print(f"interpretation: {interpret(best[1])}")
        if best[1] < 0.60:
            print("\nkappa below 0.60 -> report both judge metrics as INDICATIVE ONLY.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sample", "label", "kappa"])
    args = ap.parse_args()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    try:
        return {"sample": cmd_sample, "label": cmd_label, "kappa": cmd_kappa}[args.command](conn)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
