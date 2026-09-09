"""M01-E bounded IntervalTier preview; generated frontend types share this source."""
from pydantic import Field
from .models import WireModel,Hash,Identifier


class PreviewInterval(WireModel):
    xmin: float
    xmax: float
    text: str


class PreviewTier(WireModel):
    name: str
    intervals: list[PreviewInterval] = Field(max_length=100000)


class TextGridPreview(WireModel):
    asset_id: Identifier
    sha256: Hash
    tiers: list[PreviewTier] = Field(max_length=64)


class SpectrogramPreview(WireModel):
    sha256: Hash
    start: float
    end: float
    frequency_max: float
    x1: float
    dx: float
    y1: float
    dy: float
    width: int = Field(ge=1,le=1002)
    height: int = Field(ge=1,le=252)
    pixels_base64: str = Field(max_length=400000)
    backend: str
    parselmouth_version: str
    praat_version: str
    window_length: float
    dynamic_range: float
    preemphasis: float
    time_step: float
