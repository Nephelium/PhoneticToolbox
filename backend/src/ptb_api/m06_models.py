"""M06 versioned task boundary; scientific validation lives in the core."""
from typing import Literal
from pydantic import Field, model_validator
from .models import WireModel,Identifier,IdempotencyKey
from .acoustic_models import AcousticAssetRef
from .acoustic_batch_models import AcousticManagedFile
from .storage_policy import PolicyVersion,LEGACY_POLICY_VERSION


class M06Request(WireModel):
    schema_version: Literal['m06/1']='m06/1'
    project_id: Identifier
    idempotency_key: IdempotencyKey
    action: Literal['generate','synthesize','extract']
    parameters: AcousticAssetRef
    audio: AcousticAssetRef | None=None

    @model_validator(mode='after')
    def valid(self):
        if (self.action=='extract') != (self.audio is not None):raise ValueError('m06_audio_action_mismatch')
        return self


class M06Manifest(WireModel):
    policy_version: PolicyVersion=LEGACY_POLICY_VERSION
    kind: Literal['managed_m06_files']='managed_m06_files'
    complete: Literal[True]=True
    operation: Literal['speech_synthesis']='speech_synthesis'
    core_version: str
    # Keep historical R4 four-file results readable; new tasks emit 2 or 3 files.
    files: list[AcousticManagedFile]=Field(min_length=2,max_length=4)
