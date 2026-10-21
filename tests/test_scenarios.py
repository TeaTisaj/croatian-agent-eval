"""Feasibility gate on the eval set.

Every `confirmed` expectation is replayed against a fresh Store. If a slot we
expect the model to book is actually taken, closed, or served by a staff member
who doesn't do that service, the scenario is unwinnable and would silently
measure nothing. This test is the reason to trust the completion numbers.
"""

import pytest

from hreval.domain.store import Store, ToolError
from hreval.scenarios.schema import load

SCENARIOS = load()
CONFIRMED = [s for s in SCENARIOS if s.expected.outcome == "confirmed"]


def test_set_is_complete():
    assert len(SCENARIOS) == 36


@pytest.mark.parametrize("s", CONFIRMED, ids=[s.id for s in CONFIRMED])
def test_expected_booking_is_actually_achievable(s, now):
    store = Store(frozen_now=now)
    args = s.expected.final_args
    slots = store.available_slots(args["service_id"], args["start"][:10], staff_id=args["staff_id"])
    assert args["start"] in [x["start"] for x in slots], (
        f"{s.id}: expected slot {args['start']} is not offered by the backend"
    )
    held = store.create_hold(args["service_id"], args["staff_id"], args["start"], "Test", "000")
    confirmed = store.confirm(held.ref)
    assert confirmed.location_id == args["location_id"], f"{s.id}: location mismatch"
    assert store.terminal_state()["n_confirmed"] == 1


@pytest.mark.parametrize("s", [x for x in SCENARIOS if x.id in {"ad01", "ad04"}], ids=["ad01", "ad04"])
def test_refusal_scenarios_are_genuinely_unsatisfiable(s, now):
    store = Store(frozen_now=now)
    if s.id == "ad01":
        assert store.list_services("manikura") == []
        assert store.list_services("nokti") == []
    if s.id == "ad04":
        with pytest.raises(ToolError) as exc:
            store.available_slots("musko_sisanje", "2026-03-15")
        assert exc.value.code == "closed_that_day"


def test_english_turns_are_not_transliterated_croatian():
    """Cheap guard on the anti-translationese rule."""
    for s in SCENARIOS:
        blob = " ".join(s.turns_en).lower()
        for marker in ("moze", "molim", "hvala", "sutra", "termin"):
            assert marker not in blob.split(), f"{s.id}: Croatian leaked into the English arm"


def test_min_turns_are_plausible():
    for s in SCENARIOS:
        floor = len(s.turns_hr) + (4 if s.expected.outcome != "no_booking" else 1)
        assert s.min_turns >= len(s.turns_hr), f"{s.id}: min_turns below the number of user turns"
        assert s.min_turns <= floor + 3, f"{s.id}: min_turns={s.min_turns} looks inflated"
