import json

from hreval.domain.tools import TOOL_NAMES, dispatch, tool_specs


def _props(specs, tool):
    return next(s for s in specs if s["function"]["name"] == tool)["function"]["parameters"]


def test_both_languages_expose_identical_schemas():
    """Harness-translation rule: descriptions differ, parameter KEYS never do."""
    hr, en = tool_specs("hr"), tool_specs("en")
    assert [s["function"]["name"] for s in hr] == TOOL_NAMES
    assert [s["function"]["name"] for s in en] == TOOL_NAMES
    for tool in TOOL_NAMES:
        assert _props(hr, tool) == _props(en, tool) or (
            sorted(_props(hr, tool)["properties"]) == sorted(_props(en, tool)["properties"])
            and _props(hr, tool)["required"] == _props(en, tool)["required"]
        )


def test_descriptions_actually_differ():
    hr, en = tool_specs("hr"), tool_specs("en")
    for a, b in zip(hr, en):
        assert a["function"]["description"] != b["function"]["description"]


def test_croatian_harness_contains_no_english_descriptions():
    blob = json.dumps(tool_specs("hr"), ensure_ascii=False).lower()
    for word in (" the ", " and ", "optional ", "customer's"):
        assert word not in blob, word


def test_errors_are_localised(store):
    hr = dispatch(store, "get_available_slots", {"service_id": "fade", "date": "2026-03-15"}, "hr")
    en = dispatch(store, "get_available_slots", {"service_id": "fade", "date": "2026-03-15"}, "en")
    assert hr["error"] == en["error"] == "closed_that_day"
    assert hr["message"] != en["message"]
    assert "Salon ne radi" in hr["message"]


def test_missing_arguments_are_reported_not_repaired(store):
    out = dispatch(store, "create_booking_hold", {"service_id": "fade"}, "hr")
    assert out["error"] == "missing_arguments"
    assert store.terminal_state()["n_confirmed"] == 0


def test_unknown_tool_is_an_error(store):
    assert dispatch(store, "book_flight", {}, "en")["error"] == "unknown_tool"
