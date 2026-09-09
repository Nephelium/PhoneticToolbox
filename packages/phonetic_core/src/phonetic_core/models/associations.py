"""Parsed association values; no paths or deserialization in the scientific core."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    xmin: float
    xmax: float
    text: str


@dataclass(frozen=True)
class Tier:
    name: str
    intervals: tuple[Interval, ...]


@dataclass(frozen=True)
class AcousticAssociations:
    lip: dict | None = None
    lip_companion_start: float | None = None
    tiers: tuple[Tier, ...] = ()
