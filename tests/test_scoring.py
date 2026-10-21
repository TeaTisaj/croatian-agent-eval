"""Scorer correctness, on hand-built traces.

If the scorer is wrong, every number downstream is wrong and nothing else in
the pipeline will tell you.
"""

from hreval.scenarios.schema import load as load_scenarios
from hreval.scoring.deterministic import classify_failure, score_args, score_outcome, score_tools

BY_ID = {s.id: s for s in load_scenarios()}


def trace(tool_names, confirmed=None, stop="completed", malformed=False, unknown=False):
    return {
        "stop_reason": stop,
        "tool_calls": [{"name": n, "arguments": {}, "malformed": malformed, "unknown_tool": unknown}
                       for n in tool_names],
        "assistant_texts": [],
        "terminal_state": {"confirmed": [confirmed] if confirmed else [], "n_confirmed": 1 if confirmed else 0},
        "usage": {"n_model_calls": len(tool_names) + 1, "prompt_tokens": 0, "completion_tokens": 0},
    }


def _good_booking(s):
    return dict(s.expected.final_args)


def test_correct_booking_scores_complete():
    s = BY_ID["hp01"]
    t = trace(["search_services", "get_available_slots", "create_booking_hold", "confirm_booking"],
              _good_booking(s))
    assert score_outcome(t, s) is True
    assert score_tools(t, s)[0] is True
    assert score_args(t, s) == (4, 4)


def test_wrong_time_fails_outcome_but_scores_partial_fields():
    """The 'pola tri' failure mode: right service, right staff, wrong hour."""
    s = BY_ID["ls02"]
    booking = _good_booking(s) | {"start": "2026-03-12T15:30"}
    t = trace(["search_services", "get_available_slots", "create_booking_hold", "confirm_booking"], booking)
    assert score_outcome(t, s) is False
    assert score_args(t, s) == (3, 4)
    assert classify_failure(t, s, False, False) == "bad_arg_date"


def test_wrong_staff_is_an_entity_error():
    s = BY_ID["hp01"]
    t = trace(["search_services", "get_available_slots", "create_booking_hold", "confirm_booking"],
              _good_booking(s) | {"staff_id": "petra", "location_id": "split"})
    assert classify_failure(t, s, False, False) == "bad_arg_entity"


def test_out_of_order_confirm_breaks_tool_score():
    s = BY_ID["hp01"]
    t = trace(["search_services", "get_available_slots", "confirm_booking", "create_booking_hold"],
              _good_booking(s))
    ok, problems = score_tools(t, s)
    assert ok is False and any(p.startswith("order:") for p in problems)


def test_refusal_scenario_passes_when_nothing_is_booked():
    s = BY_ID["ad01"]
    t = trace(["search_services"])
    assert score_outcome(t, s) is True
    assert score_tools(t, s)[0] is True


def test_refusal_scenario_fails_if_it_books_anyway():
    s = BY_ID["ad01"]
    t = trace(["search_services", "create_booking_hold", "confirm_booking"], {"service_id": "x"})
    assert score_outcome(t, s) is False
    assert classify_failure(t, s, False, False) == "premature_finalize"


def test_cancellation_requires_the_tool_not_just_a_promise():
    s = BY_ID["mt05"]
    said_only = trace(["search_services", "get_available_slots", "create_booking_hold", "confirm_booking"])
    assert score_outcome(said_only, s) is False
    actually = trace(["search_services", "get_available_slots", "create_booking_hold",
                      "confirm_booking", "cancel_booking"])
    assert score_outcome(actually, s) is True


def test_loop_exhaustion_is_its_own_cause():
    s = BY_ID["hp01"]
    t = trace(["search_services"] * 8, stop="loop_exhausted")
    assert classify_failure(t, s, False, False) == "loop_exhausted"


def test_language_drift_detection():
    from hreval.scoring.deterministic import detect_drift
    assert detect_drift(["Your appointment has been confirmed for tomorrow at 11 with Marko."], "hr") is True
    assert detect_drift(["Termin je potvrđen za sutra u 11 sati kod Marka."], "hr") is False
    assert detect_drift(["Your appointment is confirmed."], "en") is False


def test_cohens_kappa_maths():
    from hreval.scoring.agreement import cohens_kappa
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    k = cohens_kappa([1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 1])
    assert 0.30 < k < 0.40          # chance-corrected, not raw agreement
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) is None   # no variance -> undefined
    assert cohens_kappa([], []) is None


def test_dev_test_split_is_deterministic_and_roughly_one_third():
    from hreval.scoring.agreement import _split_for
    keys = [f"hp{i:02d}|flagship|HR_HR|1" for i in range(300)]
    first = [_split_for(k) for k in keys]
    assert first == [_split_for(k) for k in keys]       # stable across calls
    dev_share = first.count("dev") / len(first)
    assert 0.25 < dev_share < 0.42
