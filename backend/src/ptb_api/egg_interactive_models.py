from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from .egg_models import EggTaskConfig, EggPreviewData


class EggPreviewSession(BaseModel):
    session_id: UUID
    sha256: str = Field(pattern='^[0-9a-f]{64}$')


class EggInteractiveResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    input_sha256: str = Field(pattern='^[0-9a-f]{64}$')
    sample_rate_hz: int
    sample_count: int
    config: EggTaskConfig
    selection: dict[str, float]
    preview: EggPreviewData
    psd_base64: str = Field(max_length=2_000_000)
    audio_base64: str | None = Field(default=None, max_length=32_000_000)
