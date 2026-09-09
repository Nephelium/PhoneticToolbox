"""Bounded P06 metadata contracts, shared by local and server APIs."""
from typing import Annotated, Literal
from pydantic import Field
from .models import WireModel

State = Literal['queued','running','cancel_requested','cancelled','failed','interrupted','succeeded']
Identifier = Annotated[str, Field(pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')]
IdempotencyKey = Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]{8,80}$')]


class ProbeConfig(WireModel):
    sample_count: Annotated[int, Field(ge=1,le=4096)] = 4096
    seed: Annotated[int, Field(ge=0,le=255)] = 0


class JobInput(WireModel):
    project_id: Identifier
    idempotency_key: IdempotencyKey
    operation: Literal['pipeline_check'] = 'pipeline_check'
    config: ProbeConfig = Field(default_factory=ProbeConfig)


class RetryInput(WireModel):
    idempotency_key: IdempotencyKey


class JobManifest(WireModel):
    complete: Literal[True] = True
    kind: Literal['pipeline_check_metadata'] = 'pipeline_check_metadata'
    sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    sample_count: Annotated[int, Field(ge=1,le=4096)]
    core_version: str


class JobView(WireModel):
    id: str
    project_id: str
    operation: Literal['pipeline_check'] = 'pipeline_check'
    state: State
    progress: Annotated[float, Field(ge=0,le=1)]
    generation: int
    created_at: float
    updated_at: float
    result_manifest: JobManifest | None
    error_code: str | None
    retry_of: str | None = None


class JobList(WireModel):
    jobs: list[JobView]


class JobEvent(WireModel):
    sequence: int
    state: State
    progress: float
    code: str
    created_at: float


class JobEvents(WireModel):
    events: list[JobEvent]
