"""End-to-end harness tests using a scripted provider — no API key needed.

These lock down the measurement-integrity rules: malformed calls are recorded
rather than repaired, the loop is bounded, and a hold alone never counts as a
completed booking.
"""

import pytest

from hreval.harness.agent import run_scenario
from hreval.providers import ScriptedProvider


async def _run(store, now, script, turns, lang="hr", **kw):
    return await run_scenario(
        provider=ScriptedProvider(script), model="test", user_turns=turns,
        harness_lang=lang, store=store, frozen_now=now, **kw,
    )


async def test_happy_path_books_and_confirms(store, now):
    script = [
        {"tool_calls": [{"name": "search_services", "arguments": {"query": "fade"}}]},
        {"tool_calls": [{"name": "get_available_slots",
                         "arguments": {"service_id": "fade", "date": "2026-03-11", "staff_id": "marko"}}]},
        {"tool_calls": [{"name": "create_booking_hold", "arguments": {
            "service_id": "fade", "staff_id": "marko", "start_datetime": "2026-03-11T11:00",
            "customer_name": "Ana Horvat", "customer_phone": "0915551234"}}]},
        {"tool_calls": [{"name": "confirm_booking", "arguments": {"ref": "H001"}}]},
        {"content": "Termin je potvrđen."},
    ]
    res = await _run(store, now, script, ["Trebam fade kod Marka sutra u 11."])
    assert res.stop_reason == "completed"
    assert res.terminal_state["n_confirmed"] == 1
    booking = res.terminal_state["confirmed"][0]
    assert booking["start"] == "2026-03-11T11:00"
    assert booking["staff_id"] == "marko"
    assert [c["name"] for c in res.tool_calls] == [
        "search_services", "get_available_slots", "create_booking_hold", "confirm_booking"]


async def test_hold_without_confirm_is_not_a_completion(store, now):
    script = [
        {"tool_calls": [{"name": "create_booking_hold", "arguments": {
            "service_id": "fade", "staff_id": "marko", "start_datetime": "2026-03-11T11:00",
            "customer_name": "Ana", "customer_phone": "091"}}]},
        {"content": "Je li u redu?"},
    ]
    res = await _run(store, now, script, ["Fade sutra u 11."])
    assert res.terminal_state["n_confirmed"] == 0


async def test_malformed_tool_call_is_recorded_not_repaired(store, now):
    provider = ScriptedProvider([])

    async def broken(**_kw):
        return {"content": "", "prompt_tokens": 1, "completion_tokens": 1,
                "tool_calls": [{"id": "c0", "name": "search_services", "arguments": "{not json"}],
                "raw_tool_calls": [{"id": "c0", "type": "function",
                                    "function": {"name": "search_services", "arguments": "{not json"}}]}

    provider.chat = broken
    res = await run_scenario(provider=provider, model="t", user_turns=["Bok"], harness_lang="hr",
                             store=store, frozen_now=now, max_tool_calls=2)
    assert res.tool_calls[0]["malformed"] is True
    assert res.terminal_state["n_confirmed"] == 0


async def test_hallucinated_tool_is_flagged(store, now):
    script = [{"tool_calls": [{"name": "book_flight", "arguments": {}}]}, {"content": "Ispričavam se."}]
    res = await _run(store, now, script, ["Rezerviraj mi let."])
    assert res.tool_calls[0]["unknown_tool"] is True


async def test_loop_is_bounded(store, now):
    script = [{"tool_calls": [{"name": "search_services", "arguments": {}}]} for _ in range(20)]
    res = await _run(store, now, script, ["Bok"], max_tool_calls=3)
    assert res.stop_reason == "loop_exhausted"
    assert len(res.tool_calls) == 3


async def test_multi_turn_delivers_turns_one_at_a_time(store, now):
    script = [
        {"content": "Naravno, koji datum?"},
        {"content": "U redu, koje vrijeme?"},
        {"content": "Gotovo."},
    ]
    res = await _run(store, now, script, ["Trebam šišanje.", "Sutra.", "U 11."])
    assert len(res.assistant_texts) == 3


async def test_harness_language_reaches_the_model(store, now):
    provider = ScriptedProvider([{"content": "ok"}])
    await run_scenario(provider=provider, model="t", user_turns=["Bok"], harness_lang="hr",
                       store=store, frozen_now=now)
    system = provider.calls[0]["messages"][0]["content"]
    assert "Salonu Jadran" in system and "You are" not in system
    assert "Pretraži usluge" in provider.calls[0]["tools"][0]["function"]["description"]


async def test_user_turns_wait_for_an_assistant_reply(store, now):
    """Regression: a user turn must never land straight after a tool result.

    The original loop popped a pending turn at the top of every iteration, so
    the customer's confirmation arrived before the assistant had said anything
    and the run ended before confirm_booking could be reached. That produced a
    silent 0% completion rate for every model in every language.
    """
    script = [
        {"tool_calls": [{"name": "search_services", "arguments": {"query": "fade"}}]},
        {"tool_calls": [{"name": "get_available_slots",
                         "arguments": {"service_id": "fade", "date": "2026-03-11", "staff_id": "marko"}}]},
        {"tool_calls": [{"name": "create_booking_hold", "arguments": {
            "service_id": "fade", "staff_id": "marko", "start_datetime": "2026-03-11T11:00",
            "customer_name": "Ana", "customer_phone": "091"}}]},
        {"content": "Rezervirao sam fade kod Marka sutra u 11. Potvrđujete?"},
        {"tool_calls": [{"name": "confirm_booking", "arguments": {"ref": "H001"}}]},
        {"content": "Potvrđeno."},
    ]
    res = await _run(store, now, script, ["Trebam fade kod Marka sutra u 11.", "Da, potvrđujem."])

    kinds = [(s.kind, s.payload) for s in res.steps]
    user_positions = [i for i, (k, _) in enumerate(kinds) if k == "user"]
    assert len(user_positions) == 2
    # Everything between the two user turns must include an assistant reply.
    between = [k for k, _ in kinds[user_positions[0]:user_positions[1]]]
    assert "assistant" in between, f"second user turn arrived with no assistant reply first: {between}"
    assert res.terminal_state["n_confirmed"] == 1
