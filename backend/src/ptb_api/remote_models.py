"""Single source for remote/1; no executable paths or user selected origins."""
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from .lpc_models import LpcTaskConfig
from .egg_models import EggTaskConfig
from .acoustic_models import AcousticConfigSnapshot, AcousticAssetRef


class RemoteModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Poll(RemoteModel):
    request_id: UUID
    runtime_hash: str = Field(pattern=r'^[0-9a-f]{64}$')


class Health(RemoteModel):
    runtime_hash: str = Field(pattern=r'^[0-9a-f]{64}$')


class AttemptRequest(RemoteModel):
    generation: int = Field(ge=1)


class Heartbeat(AttemptRequest):
    phase: Literal['download', 'compute', 'upload']
    node_bytes: int = Field(default=0, ge=0, le=1_000_000_000_000)


class Output(AttemptRequest):
    key: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    name: str = Field(pattern=r'^[^/\\\x00-\x1f]{1,128}$')
    size_bytes: int = Field(ge=1, le=1_000_000_000)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class Failure(AttemptRequest):
    code: Literal['network_error', 'transfer_stalled', 'node_shutdown', 'invalid_input', 'algorithm_error',
                  'missing_model', 'authentication_failed', 'cancelled', 'resource_limit']


class InputAsset(RemoteModel):
    id: UUID
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    size_bytes: int = Field(ge=1, le=1_000_000_000)
    expires_at: float
    expires_remaining_seconds: float = Field(gt=0)


class ExecutionConfig(RemoteModel):
    inputs: list[UUID] = Field(min_length=1, max_length=16)
    analysis: LpcTaskConfig | EggTaskConfig | AcousticConfigSnapshot
    max_output_bytes: int = Field(gt=0, le=1_000_000_000)


class Snapshot(RemoteModel):
    operation: Literal['lpc_analysis', 'egg_analysis', 'acoustic_analysis']
    core_version: str = Field(min_length=1, max_length=64)
    adapter_version: str = Field(min_length=1, max_length=64)
    config: ExecutionConfig
    input_refs: dict[str, AcousticAssetRef | None]

    @model_validator(mode='after')
    def matching_analysis(self):
        expected = {'lpc_analysis': LpcTaskConfig, 'egg_analysis': EggTaskConfig,
                    'acoustic_analysis': AcousticConfigSnapshot}[self.operation]
        if not isinstance(self.config.analysis, expected):
            raise ValueError('analysis model does not match operation')
        return self


class LeaseResponse(RemoteModel):
    server_time: float
    lease_until: float
    deadline: float
    lease_remaining_seconds: float = Field(gt=0)
    deadline_remaining_seconds: float = Field(gt=0)


class HealthResponse(RemoteModel):
    server_time: float
    healthy_count: int = Field(ge=0)


class UploadResponse(RemoteModel):
    upload_id: UUID
    offset: int = Field(ge=0)


class OffsetResponse(RemoteModel):
    offset: int = Field(ge=0)


class CompleteResponse(RemoteModel):
    # C treats this as an opaque P07 receipt, never as executable configuration.
    result_manifest: dict[str, JsonValue]


class FailResponse(RemoteModel):
    accepted: Literal[True] = True


class Claim(LeaseResponse):
    protocol: Literal['remote/1'] = 'remote/1'
    attempt_id: UUID
    job_id: UUID
    generation: int = Field(ge=1)
    location: str
    operation: Literal['lpc_analysis', 'egg_analysis', 'acoustic_analysis']
    runtime_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    parameter_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    font_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    input_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    snapshot: Snapshot
    inputs: list[InputAsset] = Field(min_length=1, max_length=16)
    memory_bytes: int = Field(gt=0)
    max_output_bytes: int = Field(gt=0, le=1_000_000_000)


REMOTE_MODELS = (Poll, Health, AttemptRequest, Heartbeat, Output, Failure, Claim,
                 LeaseResponse, HealthResponse, UploadResponse, OffsetResponse, CompleteResponse, FailResponse)
