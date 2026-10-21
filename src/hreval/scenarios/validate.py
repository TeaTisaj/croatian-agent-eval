"""`make validate` — structural gate on the eval set, run before any matrix."""

from __future__ import annotations

import sys
from collections import Counter

from .schema import load

TARGET_BUCKETS = {
    "happy_path": 12, "linguistic_stress": 8, "code_switching": 6,
    "multi_turn": 5, "adversarial": 5,
}
REQUIRED_AXES = {"diacritics", "inflection", "time_idiom", "code_switching",
                 "relative_dates", "register_dialect"}
MIN_PER_AXIS = 3


def main() -> int:
    scenarios = load()
    problems: list[str] = []

    if not scenarios:
        print("no scenarios yet — eval_set.jsonl is empty")
        return 1

    ids = Counter(s.id for s in scenarios)
    problems += [f"duplicate id: {i}" for i, n in ids.items() if n > 1]

    buckets = Counter(s.bucket for s in scenarios)
    for bucket, want in TARGET_BUCKETS.items():
        got = buckets.get(bucket, 0)
        if got != want:
            problems.append(f"bucket {bucket}: {got} scenarios, target {want}")

    axes = Counter(a for s in scenarios for a in s.axes)
    for axis in sorted(REQUIRED_AXES):
        if axes.get(axis, 0) < MIN_PER_AXIS:
            problems.append(f"axis {axis}: {axes.get(axis, 0)} scenarios, need >= {MIN_PER_AXIS}")

    for s in scenarios:
        if s.turns_hr == s.turns_en:
            problems.append(f"{s.id}: HR and EN turns are identical")
        if s.expected.outcome == "confirmed" and not s.expected.final_args:
            problems.append(f"{s.id}: outcome 'confirmed' but no final_args to score")

    print(f"{len(scenarios)} scenarios | buckets {dict(buckets)}")
    print(f"axis coverage {dict(axes)}")
    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nvalidate: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
