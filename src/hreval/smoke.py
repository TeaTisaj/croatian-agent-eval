"""Block 0 go/no-go.

Resolves the tier ladder to dated snapshot ids and pins them (never `-latest`,
which moves under you and destroys reproducibility), then measures achievable
throughput and extrapolates it to the full matrix. If the matrix would take
longer than the budget, we cut repeats NOW rather than discovering it at 2am.
"""

from __future__ import annotations

import asyncio
import json
import time
import tomllib
from pathlib import Path

from dotenv import load_dotenv

from .providers import MistralProvider

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "config" / "models.lock.json"
RUN_CONFIG = ROOT / "config" / "run.toml"

# The tier ladder, derived from what the account can actually see.
#
# Note: there is no `mistral-large` in the current lineup - the Medium 3 line
# superseded it. So the ladder is flagship -> mid -> edge, which maps better
# onto the real deployment question anyway ("can the cheap tier carry this?").
# Preference order per tier; the first id present in the account wins, and a
# dated snapshot always beats a moving `-latest` alias.
TIER_PATTERNS = {
    "flagship": ["mistral-medium-26", "mistral-medium-3.5", "mistral-large", "mistral-medium"],
    "mid":      ["mistral-small-26", "mistral-small"],
    "edge":     ["ministral-8b", "ministral-3b", "ministral-14b"],
}
TIERS = ("flagship", "mid", "edge")
WALL_CLOCK_BUDGET_HOURS = 6.0


def _pick(tier: str, available: list[str]) -> str | None:
    """Prefer a dated snapshot over a moving alias."""
    for pattern in TIER_PATTERNS[tier]:
        matches = [m for m in available if m.startswith(pattern)]
        if not matches:
            continue
        dated = sorted(m for m in matches if any(ch.isdigit() for ch in m.rsplit("-", 1)[-1])
                       and not m.endswith("latest"))
        return dated[-1] if dated else sorted(matches)[0]
    return None


async def main() -> int:
    load_dotenv(ROOT / ".env")
    cfg = tomllib.loads(RUN_CONFIG.read_text())
    limits, matrix = cfg["limits"], cfg["matrix"]

    provider = MistralProvider(
        rate_per_sec=limits["requests_per_second"],
        max_retries=limits["max_retries"],
    )
    try:
        available = await provider.list_models()
        print(f"{len(available)} models visible to this key\n")

        pinned: dict[str, str] = {}
        for tier in TIERS:
            chosen = _pick(tier, available)
            if chosen is None:
                print(f"  {tier:9s} -> NOT FOUND")
            else:
                pinned[tier] = chosen
                print(f"  {tier:9s} -> {chosen}")
        if len(pinned) < 2:
            print("\nFAIL: need at least two tiers to run a ladder comparison")
            return 1

        print("\nreachability + throughput:")
        latencies: dict[str, float] = {}
        for tier, model in pinned.items():
            started = time.monotonic()
            reply = await provider.chat(
                model=model,
                messages=[{"role": "user", "content": "Reci samo: OK"}],
                temperature=0.0, seed=0, max_tokens=16,
            )
            elapsed = time.monotonic() - started
            latencies[tier] = elapsed
            text = (reply["content"] or "").strip().replace("\n", " ")[:40]
            print(f"  {tier:9s} {elapsed:5.2f}s  {reply['prompt_tokens']:>4}+{reply['completion_tokens']:<3} tok  {text!r}")

        n_scenarios = 36
        n_calls = len(pinned) * len(matrix["conditions"]) * n_scenarios * matrix["repeats"] * 4
        per_call = max(max(latencies.values()), 1.0 / limits["requests_per_second"])
        hours = n_calls * per_call / limits["max_concurrency"] / 3600

        print(f"\nprojected matrix: ~{n_calls} model calls")
        print(f"  at {per_call:.2f}s/call, concurrency {limits['max_concurrency']} -> ~{hours:.1f}h wall clock")
        verdict = "GO" if hours <= WALL_CLOCK_BUDGET_HOURS else "CUT REPEATS"
        print(f"  budget {WALL_CLOCK_BUDGET_HOURS:.0f}h -> {verdict}")

        LOCK_PATH.write_text(json.dumps(
            {"resolved_at_frozen_config": True, "tiers": pinned,
             "note": "Pinned ids. Never use -latest: aliases move and break reproducibility."},
            indent=2) + "\n")
        print(f"\nwrote {LOCK_PATH.relative_to(ROOT)}")
        return 0
    finally:
        await provider.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
