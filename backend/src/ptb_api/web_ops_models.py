"""Shared manager wire inputs, without scientific or PostgreSQL imports."""
from uuid import UUID
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field

class Decision(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id: UUID
    granted_bytes: int | None = Field(default=None,ge=100_000_000,le=10_000_000_000,strict=True)
    duration: Literal['today','7d','30d'] = 'today'
    note: str = Field(default='',max_length=500)

class QuotaChange(BaseModel):
    model_config=ConfigDict(extra='forbid')
    quota_bytes: int = Field(ge=100_000_000,le=10_000_000_000,strict=True)
    owner_id: UUID | None = None
