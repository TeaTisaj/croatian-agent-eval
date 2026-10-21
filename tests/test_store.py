import pytest

from hreval.domain.store import Store, ToolError


def test_identical_calls_give_identical_state(now):
    a, b = Store(frozen_now=now), Store(frozen_now=now)
    for s in (a, b):
        s.create_hold("fade", "marko", "2026-03-11T11:00", "Ana Horvat", "0915551234")
    assert a.terminal_state() == b.terminal_state()
    assert a.available_slots("fade", "2026-03-12") == b.available_slots("fade", "2026-03-12")


def test_seeded_appointment_blocks_slot(store):
    starts = [s["start"] for s in store.available_slots("fade", "2026-03-11", staff_id="marko")]
    assert "2026-03-11T10:00" not in starts
    assert "2026-03-11T11:00" in starts


def test_sunday_is_closed(store):
    with pytest.raises(ToolError) as exc:
        store.available_slots("fade", "2026-03-15")
    assert exc.value.code == "closed_that_day"


def test_saturday_closes_early(store):
    starts = [s["start"] for s in store.available_slots("zensko_sisanje", "2026-03-14", staff_id="iva")]
    assert all(s <= "2026-03-14T13:15" for s in starts), starts[-3:]


def test_staff_competency_enforced(store):
    with pytest.raises(ToolError) as exc:
        store.create_hold("bojanje", "marko", "2026-03-11T11:00", "A", "1")
    assert exc.value.code == "staff_cannot_do_service"


def test_double_booking_rejected(store):
    store.create_hold("fade", "marko", "2026-03-11T11:00", "A", "1")
    with pytest.raises(ToolError) as exc:
        store.create_hold("fade", "marko", "2026-03-11T11:15", "B", "2")
    assert exc.value.code == "slot_taken"


def test_hold_is_not_a_confirmed_booking(store):
    store.create_hold("fade", "marko", "2026-03-11T11:00", "A", "1")
    assert store.terminal_state()["n_confirmed"] == 0


def test_cancel_frees_the_slot(store):
    held = store.create_hold("fade", "marko", "2026-03-11T11:00", "A", "1")
    store.cancel(held.ref)
    starts = [s["start"] for s in store.available_slots("fade", "2026-03-11", staff_id="marko")]
    assert "2026-03-11T11:00" in starts


def test_past_dates_rejected(store):
    with pytest.raises(ToolError) as exc:
        store.available_slots("fade", "2026-03-01")
    assert exc.value.code == "date_in_past"
