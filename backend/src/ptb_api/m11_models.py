"""M11 protocol. Runtime paths are host configuration, never task inputs."""
from typing import Literal
from pydantic import Field, model_validator
from .models import WireModel, Identifier, IdempotencyKey
from .acoustic_models import AcousticAssetRef
from .acoustic_batch_models import AcousticManagedFile
from .storage_policy import PolicyVersion, LEGACY_POLICY_VERSION


class M11Config(WireModel):
    beam: int = Field(default=100, ge=1, le=10000, strict=True)
    retry_beam: int = Field(default=400, ge=1, le=40000, strict=True)

    @model_validator(mode='after')
    def retry_constraint(self):
        # Source: V2 MFAAutoAlignmentService.run_alignment. UI additionally
        # raises retry to >= 4*beam when beam changes, as in the V2 dialog.
        if self.retry_beam <= self.beam:
            self.retry_beam = self.beam * 4
        return self


class M11CorpusItem(WireModel):
    name: str = Field(min_length=1, max_length=240)
    audio: AcousticAssetRef
    transcript: AcousticAssetRef
    transcript_format: Literal['.lab', '.txt', '.TextGrid'] = '.lab'


class M11Request(WireModel):
    schema_version: Literal['m11/1'] = 'm11/1'
    project_id: Identifier
    idempotency_key: IdempotencyKey
    runtime_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,80}$')
    model_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,80}$')
    dictionary: AcousticAssetRef | None = None
    corpus: list[M11CorpusItem] = Field(min_length=1, max_length=100)
    config: M11Config = Field(default_factory=M11Config)


class M11Manifest(WireModel):
    policy_version: PolicyVersion = LEGACY_POLICY_VERSION
    kind: Literal['managed_m11_files'] = 'managed_m11_files'
    complete: Literal[True] = True
    operation: Literal['mfa_alignment'] = 'mfa_alignment'
    core_version: str
    files: list[AcousticManagedFile] = Field(min_length=2, max_length=102)

    @model_validator(mode='after')
    def complete_set(self):
        names = [f.name for f in self.files]
        if len(set(names)) != len(names) or 'm11-provenance.json' not in names or not any(n.endswith('.TextGrid') for n in names):
            raise ValueError('m11_incomplete_output')
        return self
