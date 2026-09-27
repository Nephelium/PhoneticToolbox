"""Bounded P06 metadata contracts, shared by local and server APIs."""
from typing import Annotated, Literal
from pydantic import Field, model_validator
from .models import WireModel, Identifier, IdempotencyKey
from .storage_policy import QUOTA_BYTES, LEGACY_QUOTA_BYTES, PolicyVersion, LEGACY_POLICY_VERSION

State = Literal['queued','running','cancel_requested','cancelled','failed','interrupted','succeeded']


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


class FileConfig(WireModel):
    inputs: list[Identifier] = Field(default_factory=list, max_length=16)
    max_output_bytes: int = Field(default=16_777_216, ge=1, le=QUOTA_BYTES)
    probe_bytes: int = Field(default=16384, ge=1, le=33_554_432)
    probe_files: int = Field(default=2, ge=1, le=4)


class FileJobInput(WireModel):
    project_id: Identifier
    idempotency_key: IdempotencyKey
    operation: Literal['storage_check','archive_zip','extract_zip']
    config: FileConfig = Field(default_factory=FileConfig)

    @model_validator(mode='after')
    def valid_inputs(self):
        count = len(self.config.inputs)
        if len(set(self.config.inputs)) != count:
            raise ValueError('Duplicate input')
        if self.operation == 'archive_zip' and not count:
            raise ValueError('Archive requires inputs')
        if self.operation == 'extract_zip' and count != 1:
            raise ValueError('Extraction requires one ZIP')
        return self


class ResultFile(WireModel):
    id: Identifier
    name: str
    kind: Literal['result','archive']
    size_bytes: int = Field(ge=0,le=LEGACY_QUOTA_BYTES)
    sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    expires_at: float


class FileManifest(WireModel):
    policy_version: PolicyVersion = LEGACY_POLICY_VERSION
    complete: Literal[True] = True
    kind: Literal['managed_files'] = 'managed_files'
    files: list[ResultFile] = Field(min_length=1,max_length=16)
    core_version: str


class JobManifest(WireModel):
    complete: Literal[True] = True
    kind: Literal['pipeline_check_metadata'] = 'pipeline_check_metadata'
    sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    sample_count: Annotated[int, Field(ge=1,le=4096)]
    core_version: str


from .acoustic_batch_models import AcousticTaskManifest
from .spec2wav_models import Spec2WavManifest
from .egg_models import EggManifest
from .lpc_models import LpcManifest
from .m08_models import M08Manifest
from .m14_models import M14Manifest
from .m06_models import M06Manifest
from .m07_models import M07Manifest
from .m11_models import M11Manifest
from .m05_models import M05Manifest


class JobView(WireModel):
    id: str
    project_id: str
    operation: Literal['pipeline_check','storage_check','archive_zip','extract_zip','acoustic_analysis','textgrid_segment','spectrogram_to_audio','egg_analysis','lpc_analysis','pitch_manipulation','phonology_induction','speech_synthesis','phonation_synthesis','mfa_alignment','lip_analysis'] = 'pipeline_check'
    state: State
    progress: Annotated[float, Field(ge=0,le=1)]
    generation: int
    created_at: float
    updated_at: float
    result_manifest: JobManifest | FileManifest | AcousticTaskManifest | Spec2WavManifest | EggManifest | LpcManifest | M08Manifest | M14Manifest | M06Manifest | M07Manifest | M11Manifest | M05Manifest | None
    error_code: str | None
    retry_of: str | None = None
    waiting_reason: str | None = None


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


# Shared future result envelope; existing JobView/routes remain P06/P07 only.
from .acoustic_models import AcousticFileManifest


class ResultManifestEnvelope(WireModel):
    manifest: Annotated[JobManifest | FileManifest | AcousticFileManifest | AcousticTaskManifest | Spec2WavManifest | EggManifest | LpcManifest | M08Manifest | M14Manifest | M06Manifest | M07Manifest | M11Manifest | M05Manifest, Field(discriminator='kind')]
