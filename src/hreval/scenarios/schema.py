"""Scenario schema.

Expectations are CONSTRAINTS, not scripts (README §Expectations). Specifying an
exact ordered tool sequence would penalise legitimate alternative paths and turn
"tool correctness" into conformity-to-our-guess. So we assert only what must be
true: which tools must appear, which must not, and the orderings that are
causally necessary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

EVAL_SET = Path(__file__).with_name("eval_set.jsonl")

Bucket = Literal["happy_path", "linguistic_stress", "code_switching", "multi_turn", "adversarial"]
Axis = Literal[
    "diacritics", "inflection", "time_idiom", "code_switching",
    "relative_dates", "register_dialect", "none",
]
Outcome = Literal["confirmed", "no_booking", "cancelled"]

ALL_TOOLS = {"search_services", "get_available_slots", "create_booking_hold",
             "confirm_booking", "cancel_booking"}


class Expected(BaseModel):
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    order_constraints: list[tuple[str, str]] = Field(default_factory=list)
    final_args: dict[str, str] = Field(default_factory=dict)
    outcome: Outcome

    @field_validator("required_tools", "forbidden_tools")
    @classmethod
    def _known_tools(cls, v: list[str]) -> list[str]:
        unknown = set(v) - ALL_TOOLS
        if unknown:
            raise ValueError(f"unknown tools: {sorted(unknown)}")
        return v

    @model_validator(mode="after")
    def _coherent(self):
        clash = set(self.required_tools) & set(self.forbidden_tools)
        if clash:
            raise ValueError(f"tool both required and forbidden: {sorted(clash)}")
        if self.outcome == "confirmed" and "confirm_booking" not in self.required_tools:
            raise ValueError("outcome 'confirmed' requires confirm_booking")
        if self.outcome == "no_booking" and "confirm_booking" not in self.forbidden_tools:
            raise ValueError("outcome 'no_booking' must forbid confirm_booking")
        return self


class Scenario(BaseModel):
    id: str
    bucket: Bucket
    axes: list[Axis]
    intent_spec: str = Field(description="Language-neutral statement of what the customer wants")
    turns_hr: list[str] = Field(min_length=1)
    turns_en: list[str] = Field(min_length=1)
    expected: Expected
    min_turns: int = Field(ge=1, description="Hand-derived minimum model calls; step efficiency is meaningless if wrong")
    notes: str = ""

    @model_validator(mode="after")
    def _parallel(self):
        if len(self.turns_hr) != len(self.turns_en):
            raise ValueError(f"{self.id}: HR and EN must have the same number of turns")
        return self


def load(path: Path = EVAL_SET) -> list[Scenario]:
    if not path.exists():
        return []
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            out.append(Scenario.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path.name} line {i}: {exc}") from None
    return out
