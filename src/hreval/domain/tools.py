"""The five booking tools, in two parallel language variants.

Harness-translation rule (README §Method): the Croatian variant translates tool
DESCRIPTIONS, enum display labels and error messages. JSON parameter KEYS stay
English identifiers in both variants. Translating keys would confound natural
language comprehension with schema-identifier parsing, and would force the
scorer to be language-aware.
"""

from __future__ import annotations

from typing import Any, Literal

from .store import Store, ToolError

Lang = Literal["hr", "en"]

TOOL_NAMES = [
    "search_services",
    "get_available_slots",
    "create_booking_hold",
    "confirm_booking",
    "cancel_booking",
]

WRITE_TOOLS = {"create_booking_hold", "confirm_booking", "cancel_booking"}

# ── tool descriptions ─────────────────────────────────────────────────────

_DESC: dict[str, dict[Lang, str]] = {
    "search_services": {
        "en": "Look up services the salon offers. Returns id, name, duration and price. "
              "Call this before booking so you use a real service id.",
        "hr": "Pretraži usluge koje salon nudi. Vraća id, naziv, trajanje i cijenu. "
              "Pozovi prije rezervacije kako bi koristio postojeći id usluge.",
    },
    "get_available_slots": {
        "en": "List free appointment slots for a service on a given date. "
              "Optionally restrict to one staff member or one location.",
        "hr": "Prikaži slobodne termine za uslugu na određeni datum. "
              "Po želji ograniči na jednog djelatnika ili jednu lokaciju.",
    },
    "create_booking_hold": {
        "en": "Reserve a slot temporarily and return a reference. This does NOT finalise "
              "the booking — read the details back to the customer and then call confirm_booking.",
        "hr": "Privremeno rezerviraj termin i vrati referencu. Ovime rezervacija NIJE dovršena — "
              "ponovi detalje korisniku pa pozovi confirm_booking.",
    },
    "confirm_booking": {
        "en": "Finalise a held booking. Call only after the customer has explicitly agreed "
              "to the date, time, service and staff member.",
        "hr": "Dovrši privremeno rezervirani termin. Pozovi tek nakon što je korisnik izričito "
              "potvrdio datum, vrijeme, uslugu i djelatnika.",
    },
    "cancel_booking": {
        "en": "Cancel a held or confirmed booking by its reference.",
        "hr": "Otkaži privremeni ili potvrđeni termin prema referenci.",
    },
}

_PARAM: dict[str, dict[Lang, str]] = {
    "query": {"en": "Optional search text, e.g. 'haircut'.", "hr": "Neobavezan tekst pretrage, npr. 'šišanje'."},
    "service_id": {"en": "Service id from search_services.", "hr": "Id usluge dobiven iz search_services."},
    "date": {"en": "Date as YYYY-MM-DD.", "hr": "Datum u obliku YYYY-MM-DD."},
    "staff_id": {"en": "Optional staff id.", "hr": "Neobavezan id djelatnika."},
    "location_id": {"en": "Optional location id.", "hr": "Neobavezan id lokacije."},
    "start_datetime": {
        "en": "Appointment start as YYYY-MM-DDTHH:MM, taken from an available slot.",
        "hr": "Početak termina u obliku YYYY-MM-DDTHH:MM, preuzet iz slobodnog termina.",
    },
    "customer_name": {"en": "Customer's full name.", "hr": "Ime i prezime korisnika."},
    "customer_phone": {"en": "Customer's phone number.", "hr": "Broj telefona korisnika."},
    "ref": {"en": "Booking reference returned by create_booking_hold.", "hr": "Referenca koju je vratio create_booking_hold."},
}

# ── error messages (part of the harness, hence localised) ─────────────────

_ERRORS: dict[str, dict[Lang, str]] = {
    "unknown_service":        {"en": "No such service: {value}.",                       "hr": "Ne postoji usluga: {value}."},
    "unknown_staff":          {"en": "No such staff member: {value}.",                  "hr": "Ne postoji djelatnik: {value}."},
    "unknown_location":       {"en": "No such location: {value}.",                      "hr": "Ne postoji lokacija: {value}."},
    "unknown_reference":      {"en": "No booking with reference {value}.",              "hr": "Ne postoji rezervacija s referencom {value}."},
    "bad_date_format":        {"en": "Date must be YYYY-MM-DD, got {value}.",           "hr": "Datum mora biti YYYY-MM-DD, dobiveno {value}."},
    "bad_datetime_format":    {"en": "Start must be YYYY-MM-DDTHH:MM, got {value}.",    "hr": "Početak mora biti YYYY-MM-DDTHH:MM, dobiveno {value}."},
    "date_in_past":           {"en": "That date is in the past: {value}.",              "hr": "Taj datum je u prošlosti: {value}."},
    "closed_that_day":        {"en": "The salon is closed on {value}.",                 "hr": "Salon ne radi {value}."},
    "outside_opening_hours":  {"en": "{value} falls outside opening hours.",            "hr": "{value} je izvan radnog vremena."},
    "slot_taken":             {"en": "That slot is no longer free: {value}.",           "hr": "Taj termin više nije slobodan: {value}."},
    "staff_cannot_do_service":{"en": "Staff {staff} does not perform {service}.",       "hr": "Djelatnik {staff} ne pruža uslugu {service}."},
    "no_staff_for_service":   {"en": "No staff member performs {service}.",             "hr": "Nijedan djelatnik ne pruža uslugu {service}."},
    "already_confirmed":      {"en": "Booking {value} is already confirmed.",           "hr": "Rezervacija {value} je već potvrđena."},
    "already_cancelled":      {"en": "Booking {value} is already cancelled.",           "hr": "Rezervacija {value} je već otkazana."},
}


def render_error(err: ToolError, lang: Lang) -> dict:
    template = _ERRORS.get(err.code, {}).get(lang, err.code)
    return {"error": err.code, "message": template.format(**err.params)}


def _p(name: str, lang: Lang, typ: str = "string") -> dict:
    return {"type": typ, "description": _PARAM[name][lang]}


def tool_specs(lang: Lang) -> list[dict]:
    """OpenAI-compatible function specs, as Mistral's API expects them."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_services",
                "description": _DESC["search_services"][lang],
                "parameters": {
                    "type": "object",
                    "properties": {"query": _p("query", lang)},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_available_slots",
                "description": _DESC["get_available_slots"][lang],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service_id": _p("service_id", lang),
                        "date": _p("date", lang),
                        "staff_id": _p("staff_id", lang),
                        "location_id": _p("location_id", lang),
                    },
                    "required": ["service_id", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_booking_hold",
                "description": _DESC["create_booking_hold"][lang],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service_id": _p("service_id", lang),
                        "staff_id": _p("staff_id", lang),
                        "start_datetime": _p("start_datetime", lang),
                        "customer_name": _p("customer_name", lang),
                        "customer_phone": _p("customer_phone", lang),
                    },
                    "required": ["service_id", "staff_id", "start_datetime", "customer_name", "customer_phone"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "confirm_booking",
                "description": _DESC["confirm_booking"][lang],
                "parameters": {
                    "type": "object",
                    "properties": {"ref": _p("ref", lang)},
                    "required": ["ref"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_booking",
                "description": _DESC["cancel_booking"][lang],
                "parameters": {
                    "type": "object",
                    "properties": {"ref": _p("ref", lang)},
                    "required": ["ref"],
                },
            },
        },
    ]


def dispatch(store: Store, name: str, args: dict[str, Any], lang: Lang) -> dict:
    """Execute one tool call. Never repairs bad input — see measurement rule 1."""
    try:
        if name == "search_services":
            return {"services": store.list_services(args.get("query"))}
        if name == "get_available_slots":
            if "service_id" not in args or "date" not in args:
                raise ToolError("bad_date_format", value=str(args))
            slots = store.available_slots(
                args["service_id"], args["date"], args.get("staff_id"), args.get("location_id")
            )
            return {"slots": slots[:20], "n_total": len(slots)}
        if name == "create_booking_hold":
            missing = [k for k in ("service_id", "staff_id", "start_datetime", "customer_name", "customer_phone") if not args.get(k)]
            if missing:
                return {"error": "missing_arguments", "message": f"Missing: {', '.join(missing)}"}
            booking = store.create_hold(
                args["service_id"], args["staff_id"], args["start_datetime"],
                args["customer_name"], args["customer_phone"],
            )
            return {"hold": booking.as_dict()}
        if name == "confirm_booking":
            if not args.get("ref"):
                return {"error": "missing_arguments", "message": "Missing: ref"}
            return {"booking": store.confirm(args["ref"]).as_dict()}
        if name == "cancel_booking":
            if not args.get("ref"):
                return {"error": "missing_arguments", "message": "Missing: ref"}
            return {"booking": store.cancel(args["ref"]).as_dict()}
    except ToolError as err:
        return render_error(err, lang)
    return {"error": "unknown_tool", "message": f"No tool named {name}"}
