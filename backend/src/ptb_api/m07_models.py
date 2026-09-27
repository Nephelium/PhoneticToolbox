"""M07/1: finite parameters, explicit units, no client paths."""
import math
from typing import Literal
from pydantic import Field, model_validator
from .models import WireModel, Identifier, IdempotencyKey
from .acoustic_models import AcousticAssetRef
from .acoustic_batch_models import AcousticManagedFile
from .storage_policy import PolicyVersion, LEGACY_POLICY_VERSION

class M07Analysis(WireModel):
    target_sample_rate: Literal[11025]=11025
    f0_backend: Literal['parselmouth','reaper']='parselmouth'
    min_f0_hz: float=Field(default=50,ge=20,le=500)
    max_f0_hz: float=Field(default=300,ge=50,le=1000)
    f0_frame_interval_ms: float=Field(default=1,ge=.5,le=20)
    frame_length: int=Field(default=128,ge=32,le=1024)
    frame_shift: int=Field(default=32,ge=8,le=512)
    lpc_order: int=Field(default=20,ge=4,le=60)
    preemphasis: float=Field(default=.98,ge=0,lt=1)
    window_name: Literal['hamming','hann','blackman','rectangular']='hamming'
    negative_peak_threshold: float=Field(default=-.005,ge=-.2,le=0)
    pulse_inner_periods: float=Field(default=.5,ge=.1,le=1.2)
    pulse_outer_periods: float=Field(default=1.5,ge=.6,le=3)
    trim_silence: bool=True
    silence_threshold_db: float=Field(default=-45,ge=-80,le=-10)
    silence_padding_ms: float=Field(default=8,ge=0,le=80)
    voiced_margin_ms: float=Field(default=30,ge=0,le=120)
    @model_validator(mode='after')
    def ranges(self):
        if self.min_f0_hz>=self.max_f0_hz or self.frame_shift>=self.frame_length or self.lpc_order>=self.frame_length or self.pulse_inner_periods>=self.pulse_outer_periods:raise ValueError('m07_invalid_analysis_parameters')
        return self

class M07Generation(WireModel):
    step_count: int=Field(default=9,ge=2,le=50)
    energy_match: bool=True
    normalize_to_source: bool=True
    output_peak_limit: float=Field(default=.98,gt=0,le=1)

class M07Controls(WireModel):
    axis: list[float]=Field(min_length=2,max_length=200)
    source: list[float]=Field(min_length=2,max_length=200)
    target: list[float]=Field(min_length=2,max_length=200)
    @model_validator(mode='after')
    def valid(self):
        if len(self.axis)!=len(self.source) or len(self.axis)!=len(self.target):raise ValueError('m07_invalid_controls')
        if not all(math.isfinite(v) for v in self.axis+self.source+self.target):raise ValueError('m07_nonfinite_parameter')
        if self.axis[0]<0 or any(b<=a for a,b in zip(self.axis,self.axis[1:])):raise ValueError('m07_invalid_time_order')
        if any(v<0 or v>1000 for v in self.source+self.target) or not any(self.source) or not any(self.target):raise ValueError('m07_invalid_f0')
        return self

class M07Request(WireModel):
    schema_version: Literal['m07/1']='m07/1'
    project_id: Identifier
    idempotency_key: IdempotencyKey
    action: Literal['analyze','apply','generate']
    source: AcousticAssetRef
    target: AcousticAssetRef
    analysis: M07Analysis=Field(default_factory=M07Analysis)
    generation: M07Generation=Field(default_factory=M07Generation)
    analysis_job_id: Identifier | None=None
    alignment: Literal['normalize','onset']='normalize'
    point_count: int=Field(default=21,ge=20,le=200)
    controls: M07Controls | None=None
    continuum_type: Literal[1,2,3]=2
    reverse_direction: bool=False
    batch_id: Identifier | None=None
    batch_group_count: Literal[1,6]=1
    batch_group_index: int=Field(default=0,ge=0,le=5)
    @model_validator(mode='after')
    def dependencies(self):
        if self.batch_group_index>=self.batch_group_count:raise ValueError('m07_invalid_batch_group')
        if (self.action!='analyze')!=(self.analysis_job_id is not None):raise ValueError('m07_analysis_required')
        if (self.action=='apply')!=(self.controls is not None):raise ValueError('m07_controls_required')
        if self.controls and self.controls.axis[-1]>(100 if self.alignment=='normalize' else 10000):raise ValueError('m07_invalid_time_order')
        return self

class M07Manifest(WireModel):
    policy_version: PolicyVersion=LEGACY_POLICY_VERSION
    kind: Literal['managed_m07_files']='managed_m07_files'
    complete: Literal[True]=True
    operation: Literal['phonation_synthesis']='phonation_synthesis'
    core_version: str
    files: list[AcousticManagedFile]=Field(min_length=2,max_length=55)
