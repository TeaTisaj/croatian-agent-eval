"""Deterministic in-memory booking backend.

Every run gets its OWN Store instance (measurement-integrity rule 5). There is
no randomness and no wall clock anywhere in here: given the same calls in the
same order, the same state comes out. That is what makes `sutra` and
"no slot available" reproducible across 972 runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

CATALOG_PATH = Path(__file__).with_name("catalog.json")
SLOT_GRANULARITY_MIN = 15


class ToolError(Exception):
    """Structured, language-neutral failure.

    Carries a code and parameters, never a human-readable string: the message
    is rendered by the harness layer in the harness language, so that error
    text does not leak English into the HR_HR condition.
    """

    def __init__(self, code: str, **params):
        super().__init__(code)
        self.code = code
        self.params = params


@dataclass(frozen=True)
class Booking:
    ref: str
    service_id: str
    staff_id: str
    location_id: str
    start: datetime
    duration_min: int
    customer_name: str
    customer_phone: str
    status: str  # "held" | "confirmed" | "cancelled"

    def as_dict(self) -> dict:
        return {
            "ref": self.ref,
            "service_id": self.service_id,
            "staff_id": self.staff_id,
            "location_id": self.location_id,
            "start": self.start.strftime("%Y-%m-%dT%H:%M"),
            "duration_min": self.duration_min,
            "customer_name": self.customer_name,
            "customer_phone": self.customer_phone,
            "status": self.status,
        }


@dataclass
class Store:
    frozen_now: datetime
    catalog: dict = field(default_factory=lambda: json.loads(CATALOG_PATH.read_text()))
    bookings: dict[str, Booking] = field(default_factory=dict)
    _busy: list[tuple[str, datetime, datetime]] = field(default_factory=list)
    _counter: int = 0

    def __post_init__(self) -> None:
        for appt in self.catalog["seeded_appointments"]:
            start = datetime.fromisoformat(appt["start"]).replace(tzinfo=self.frozen_now.tzinfo)
            self._busy.append((appt["staff_id"], start, start + timedelta(minutes=appt["duration_min"])))

    # ── lookups ───────────────────────────────────────────────────────────

    def service(self, service_id: str) -> dict:
        for s in self.catalog["services"]:
            if s["id"] == service_id:
                return s
        raise ToolError("unknown_service", value=service_id)

    def staff_member(self, staff_id: str) -> dict:
        for s in self.catalog["staff"]:
            if s["id"] == staff_id:
                return s
        raise ToolError("unknown_staff", value=staff_id)

    def location(self, location_id: str) -> dict:
        for loc in self.catalog["locations"]:
            if loc["id"] == location_id:
                return loc
        raise ToolError("unknown_location", value=location_id)

    def list_services(self, query: str | None = None) -> list[dict]:
        out = self.catalog["services"]
        if query:
            q = _fold(query)
            out = [s for s in out if q in _fold(s["name_hr"]) or q in _fold(s["name_en"]) or q in s["id"]]
        return out

    def list_staff(self, location_id: str | None = None, service_id: str | None = None) -> list[dict]:
        out = self.catalog["staff"]
        if location_id:
            self.location(location_id)
            out = [s for s in out if s["location_id"] == location_id]
        if service_id:
            self.service(service_id)
            out = [s for s in out if service_id in s["services"]]
        return out

    # ── availability ──────────────────────────────────────────────────────

    def available_slots(
        self,
        service_id: str,
        date_str: str,
        staff_id: str | None = None,
        location_id: str | None = None,
    ) -> list[dict]:
        service = self.service(service_id)
        try:
            day = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise ToolError("bad_date_format", value=date_str) from None
        if day < self.frozen_now.date():
            raise ToolError("date_in_past", value=date_str)

        hours = self.catalog["hours"][str(day.weekday())]
        if hours is None:
            raise ToolError("closed_that_day", value=date_str)

        candidates = self.list_staff(location_id=location_id, service_id=service_id)
        if staff_id:
            member = self.staff_member(staff_id)
            if service_id not in member["services"]:
                raise ToolError("staff_cannot_do_service", staff=staff_id, service=service_id)
            candidates = [m for m in candidates if m["id"] == staff_id]
        if not candidates:
            raise ToolError("no_staff_for_service", service=service_id)

        tz = self.frozen_now.tzinfo
        open_at = datetime.combine(day, datetime.strptime(hours[0], "%H:%M").time(), tzinfo=tz)
        close_at = datetime.combine(day, datetime.strptime(hours[1], "%H:%M").time(), tzinfo=tz)
        duration = timedelta(minutes=service["duration_min"])

        slots: list[dict] = []
        for member in candidates:
            cursor = open_at
            while cursor + duration <= close_at:
                if cursor >= self.frozen_now and self._is_free(member["id"], cursor, cursor + duration):
                    slots.append(
                        {
                            "start": cursor.strftime("%Y-%m-%dT%H:%M"),
                            "staff_id": member["id"],
                            "staff_name": member["name"],
                            "location_id": member["location_id"],
                        }
                    )
                cursor += timedelta(minutes=SLOT_GRANULARITY_MIN)
        slots.sort(key=lambda s: (s["start"], s["staff_id"]))
        return slots

    def _is_free(self, staff_id: str, start: datetime, end: datetime) -> bool:
        for busy_staff, busy_start, busy_end in self._busy:
            if busy_staff == staff_id and start < busy_end and busy_start < end:
                return False
        return True

    # ── mutations ─────────────────────────────────────────────────────────

    def create_hold(
        self,
        service_id: str,
        staff_id: str,
        start_str: str,
        customer_name: str,
        customer_phone: str,
    ) -> Booking:
        service = self.service(service_id)
        member = self.staff_member(staff_id)
        if service_id not in member["services"]:
            raise ToolError("staff_cannot_do_service", staff=staff_id, service=service_id)
        try:
            start = datetime.fromisoformat(start_str)
        except ValueError:
            raise ToolError("bad_datetime_format", value=start_str) from None
        if start.tzinfo is None:
            start = start.replace(tzinfo=self.frozen_now.tzinfo)
        if start < self.frozen_now:
            raise ToolError("date_in_past", value=start_str)

        hours = self.catalog["hours"][str(start.weekday())]
        if hours is None:
            raise ToolError("closed_that_day", value=start_str)
        end = start + timedelta(minutes=service["duration_min"])
        tz = self.frozen_now.tzinfo
        close_at = datetime.combine(start.date(), datetime.strptime(hours[1], "%H:%M").time(), tzinfo=tz)
        open_at = datetime.combine(start.date(), datetime.strptime(hours[0], "%H:%M").time(), tzinfo=tz)
        if start < open_at or end > close_at:
            raise ToolError("outside_opening_hours", value=start_str)
        if not self._is_free(staff_id, start, end):
            raise ToolError("slot_taken", value=start_str)

        self._counter += 1
        ref = f"H{self._counter:03d}"
        booking = Booking(
            ref=ref,
            service_id=service_id,
            staff_id=staff_id,
            location_id=member["location_id"],
            start=start,
            duration_min=service["duration_min"],
            customer_name=customer_name,
            customer_phone=customer_phone,
            status="held",
        )
        self.bookings[ref] = booking
        self._busy.append((staff_id, start, end))
        return booking

    def confirm(self, ref: str) -> Booking:
        booking = self.bookings.get(ref)
        if booking is None:
            raise ToolError("unknown_reference", value=ref)
        if booking.status == "confirmed":
            raise ToolError("already_confirmed", value=ref)
        if booking.status == "cancelled":
            raise ToolError("already_cancelled", value=ref)
        confirmed = Booking(**{**booking.__dict__, "status": "confirmed"})
        self.bookings[ref] = confirmed
        return confirmed

    def cancel(self, ref: str) -> Booking:
        booking = self.bookings.get(ref)
        if booking is None:
            raise ToolError("unknown_reference", value=ref)
        if booking.status == "cancelled":
            raise ToolError("already_cancelled", value=ref)
        end = booking.start + timedelta(minutes=booking.duration_min)
        self._busy = [b for b in self._busy if b != (booking.staff_id, booking.start, end)]
        cancelled = Booking(**{**booking.__dict__, "status": "cancelled"})
        self.bookings[ref] = cancelled
        return cancelled

    # ── terminal state, for outcome scoring ───────────────────────────────

    def terminal_state(self) -> dict:
        confirmed = [b.as_dict() for b in self.bookings.values() if b.status == "confirmed"]
        confirmed.sort(key=lambda b: b["ref"])
        return {"confirmed": confirmed, "n_confirmed": len(confirmed)}


_FOLD = str.maketrans("čćšžđČĆŠŽĐ", "ccszdccszd")


def _fold(text: str) -> str:
    """Diacritic- and case-insensitive fold.

    Deliberately applied to CATALOGUE SEARCH ONLY. The model must still produce
    a correct service_id; this just means a customer typing "sisanje" can be
    found, which is how a real Croatian booking system has to behave.
    """
    return text.translate(_FOLD).lower().strip()
