"""Versioned image reconstruction request; no new persistence schema."""
from typing import Literal
from pydantic import Field,model_validator
from .models import WireModel,Identifier,IdempotencyKey
from .acoustic_models import AcousticAssetRef
from .acoustic_batch_models import AcousticManagedFile


class ImagePoint(WireModel):
    x:float=Field(ge=0,le=1)
    y:float=Field(ge=0,le=1)


class Spec2WavConfig(WireModel):
    time_start:float=Field(default=0,ge=0,le=86400)
    time_end:float=Field(default=1,gt=0,le=86430)
    freq_start:float=Field(default=0,ge=0,lt=48000)
    freq_end:float=Field(default=11025,ge=1,le=48000)
    min_db:float=Field(default=-30,ge=-160,le=20)
    max_db:float=Field(default=0,ge=-160,le=20)
    win_length_ms:float=Field(default=10,gt=0,le=1000)
    n_iter:int=Field(default=32,ge=1,le=128)
    target_sr:Literal[0,8000,16000,22050,24000,32000,44100,48000,96000]=44100
    seed:int=Field(default=0,ge=0,le=4294967295)
    corners:list[ImagePoint] | None=Field(default=None,min_length=4,max_length=4)

    @model_validator(mode='after')
    def ranges(self):
        if not (0<self.time_end-self.time_start<=30 and self.freq_start<self.freq_end and self.min_db<self.max_db):raise ValueError('Invalid calibration range')
        return self


class Spec2WavRequest(WireModel):
    schema_version:Literal['m09/1']='m09/1'
    project_id:Identifier
    idempotency_key:IdempotencyKey
    image:AcousticAssetRef
    config:Spec2WavConfig


from .storage_policy import PolicyVersion, LEGACY_POLICY_VERSION


class Spec2WavManifest(WireModel):
    policy_version: PolicyVersion = LEGACY_POLICY_VERSION
    kind:Literal['managed_spec2wav_files']='managed_spec2wav_files'
    complete:Literal[True]=True
    operation:Literal['spectrogram_to_audio']='spectrogram_to_audio'
    core_version:str
    files:list[AcousticManagedFile]=Field(min_length=4,max_length=4)

    @model_validator(mode='after')
    def complete_set(self):
        if sorted(f.name for f in self.files)!=['calibrated.png','reconstructed.png','reconstructed.wav','reconstruction.ptb.json'] or len({f.id for f in self.files})!=4 or sum(f.size_bytes for f in self.files)>64_000_000:raise ValueError('Incomplete reconstruction')
        return self
