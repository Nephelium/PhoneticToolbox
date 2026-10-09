"""M01-D shared wire contracts, not an enabled job operation.

source_ids: SRC-PRAAT, SRC-REAPER, SRC-IRAPT, SRC-WMPC, SRC-VOICESAUCE.
Scientific catalog labels are reused; no computation or file access at import.
"""
import hashlib
import json
import math
from typing import Annotated, Literal
from pydantic import Field, field_validator, model_validator, model_serializer
from .acoustic_extended import AcousticExtended
from phonetic_core.catalog import PARAMETER_MAPPING
from .models import WireModel, Hash, Identifier, IdempotencyKey
from .storage_policy import (QUOTA_BYTES, RETENTION_SECONDS, LEGACY_QUOTA_BYTES,
                             LEGACY_POLICY_VERSION, POLICY_VERSION, PolicyVersion, retention_seconds)

ParameterKey = Literal[tuple(PARAMETER_MAPPING)]
OutputKey = Literal[tuple(PARAMETER_MAPPING)+('SOE_pF0','SOE_rF0')]
FiniteTime = Annotated[float, Field(ge=0)]
Code = Annotated[str, Field(pattern=r'^[a-z][a-z0-9_]{0,79}$')]
SourceId = Annotated[str, Field(pattern=r'^[A-Z][A-Z0-9_-]{0,79}$')]
MAX_CELLS = 200_000
MAX_ROWS = 100_000
SCHEMA_VERSION = 'm01/1'


class AcousticSettings(WireModel):
    silence_threshold: float = Field(default=.03,ge=0,le=1)
    energy_window_ms: float = Field(default=40.,ge=1,le=1000)
    frameshift_ms: float = Field(default=5.,ge=.1,le=1000)
    windowsize_ms: float = Field(default=40.,ge=1,le=1000)
    smooth_win_size: int = Field(default=10,ge=1,le=100)
    lip_smooth_win_size: int = Field(default=0,ge=0,le=100)
    only_voiced: bool = True
    n_periods: int = Field(default=3,ge=1,le=100)
    num_formants: int = Field(default=5,ge=3,le=10)
    max_formant: float = Field(default=6000.,gt=0,le=10000)
    min_f0: float = Field(default=30.,ge=10,le=1000)
    max_f0: float = Field(default=800.,ge=50,le=2000)
    reaper_hilbert: bool = True
    reaper_no_highpass: bool = False

    @model_validator(mode='after')
    def f0_order(self):
        if self.min_f0>=self.max_f0:raise ValueError('min_f0 must be below max_f0')
        return self


class AcousticSelection(WireModel):
    mode: Literal['catalog','legacy_service'] = 'catalog'
    keys: list[ParameterKey] = Field(default_factory=lambda:list(PARAMETER_MAPPING),max_length=80)

    @model_validator(mode='after')
    def valid_selection(self):
        if len(self.keys)!=len(set(self.keys)):raise ValueError('Duplicate parameter key')
        if self.mode=='catalog' and not self.keys:raise ValueError('Empty catalog selection')
        if self.mode=='legacy_service' and self.keys:raise ValueError('Legacy service mode requires explicit empty keys')
        return self


class AcousticBackendPolicy(WireModel):
    reaper: Literal['native_required','native_then_python','python_only','disabled'] = 'native_then_python'
    wm_f0: Literal['irapt_then_praat'] = 'irapt_then_praat'


class AcousticConfigSnapshot(WireModel):
    settings: AcousticSettings = Field(default_factory=AcousticSettings)
    selection: AcousticSelection = Field(default_factory=AcousticSelection)
    backend_policy: AcousticBackendPolicy = Field(default_factory=AcousticBackendPolicy)
    extended: AcousticExtended | None = None

    @model_serializer(mode='wrap')
    def legacy_shape(self,handler):
        data=handler(self)
        if self.extended is None:data.pop('extended',None)
        return data


def config_digest(snapshot):
    raw=json.dumps(snapshot.model_dump(mode='json'),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class AcousticAssetRef(WireModel):
    asset_id: Identifier
    sha256: Hash


class AcousticInputs(WireModel):
    audio: AcousticAssetRef
    textgrid: AcousticAssetRef | None = None
    lip: AcousticAssetRef | None = None

    @model_validator(mode='after')
    def distinct_resources(self):
        ids=[v.asset_id for v in (self.audio,self.textgrid,self.lip) if v is not None]
        if len(set(ids))!=len(ids):raise ValueError('One resource cannot have multiple input roles')
        return self


class AcousticRequest(WireModel):
    schema_version: Literal['m01/1'] = SCHEMA_VERSION
    operation: Literal['acoustic_analysis'] = 'acoustic_analysis'
    project_id: Identifier
    idempotency_key: IdempotencyKey
    inputs: AcousticInputs
    config: AcousticConfigSnapshot = Field(default_factory=AcousticConfigSnapshot)


class AcousticInputSnapshot(AcousticAssetRef):
    role: Literal['audio','textgrid','lip']
    expires_at: FiniteTime | None


class AcousticDecodedAudio(WireModel):
    sample_rate_hz: int = Field(gt=0,le=384000)
    channels: int = Field(ge=1,le=8)
    sample_count: int = Field(ge=0,le=2_000_000,description='Frames per channel, never the flattened channel sample count')
    sample_dtype: Literal['uint8','int16','int32','float32','float64']

    @model_validator(mode='after')
    def allocation_bound(self):
        if self.channels*self.sample_count>2_000_000:raise ValueError('Decoded sample budget exceeded')
        return self


class AcousticBackendObservation(WireModel):
    stage: Literal['reaper','wm_f0']
    actual: Literal['native_reaper','reaper_python','irapt1','praat_fallback','unavailable','disabled']
    reason: Code | None = None
    resource_sha256: Hash | None = None

    @model_validator(mode='after')
    def backend_identity(self):
        allowed={'reaper':{'native_reaper','reaper_python','unavailable','disabled'},
                 'wm_f0':{'irapt1','praat_fallback','unavailable'}}
        if self.actual not in allowed[self.stage]:raise ValueError('Backend does not implement this stage')
        if (self.actual=='native_reaper')!=(self.resource_sha256 is not None):raise ValueError('Only actual native execution carries a binary hash')
        if self.actual in ('unavailable','praat_fallback') and self.reason is None:raise ValueError('Missing backend reason')
        return self


class AcousticMetadata(WireModel):
    computation_revision: Literal['acoustic/1','acoustic/2'] = 'acoustic/1'
    schema_version: Literal['m01/1'] = SCHEMA_VERSION
    algorithm_id: Literal['m01.parameter_estimation'] = 'm01.parameter_estimation'
    algorithm_version: Literal['legacy-numeric/1'] = 'legacy-numeric/1'
    adapter_version: Literal['m01-adapter/1'] = 'm01-adapter/1'
    core_version: str = Field(min_length=1,max_length=80)
    project_id: Identifier
    inputs: list[AcousticInputSnapshot] = Field(min_length=1,max_length=3)
    decoded: AcousticDecodedAudio
    config: AcousticConfigSnapshot
    config_sha256: Hash
    source_ids: list[SourceId] = Field(min_length=1,max_length=32)
    backends: list[AcousticBackendObservation] = Field(max_length=2)

    @model_validator(mode='after')
    def consistent_metadata(self):
        roles=[i.role for i in self.inputs]
        if roles.count('audio')!=1 or len(set(roles))!=len(roles):raise ValueError('Invalid input roles')
        if len({i.asset_id for i in self.inputs})!=len(self.inputs):raise ValueError('Duplicate input ID')
        if self.config_sha256!=config_digest(self.config):raise ValueError('Configuration digest mismatch')
        if len(set(self.source_ids))!=len(self.source_ids):raise ValueError('Duplicate source ID')
        if len({b.stage for b in self.backends})!=len(self.backends):raise ValueError('Ambiguous backend observations')
        for b in self.backends:
            if b.stage=='reaper':
                allowed={'native_required':{'native_reaper','unavailable'},'python_only':{'reaper_python','unavailable'},
                         'disabled':{'disabled'},'native_then_python':{'native_reaper','reaper_python','unavailable'}}
                if b.actual not in allowed[self.config.backend_policy.reaper]:raise ValueError('Observed backend violates requested policy')
        return self


def parameter_unit(key):
    """Descriptions reproduced from the M01-A audit; never invent calibration."""
    if key in ('pF0','rF0','pF1','pF2','pF3','pF4','pB1','pB2','pB3','pB4'):return 'Hz'
    if key.startswith(('CPP_','HNR')):return 'dB'
    if key=='Intensity':return 'dB_legacy_amplitude_reference_not_measured_SPL'
    if key.startswith('SHR_'):return 'amplitude_ratio'
    if key.startswith('SpectralSlope_'):return 'dB_per_decade'
    if key.startswith(('Jitter_','Shimmer_')):return 'percent'
    if key.startswith('Lip'):return 'unknown_without_input_metadata'
    if key in ('SOE_pF0','SOE_rF0'):return 'normalized_ZFF_difference'
    if key in PARAMETER_MAPPING:return 'dB_uncalibrated_magnitude_or_difference'
    raise ValueError('Unknown parameter')


class AcousticNumericColumn(WireModel):
    key: OutputKey
    label: str = Field(min_length=1,max_length=180)
    unit: str = Field(min_length=1,max_length=100)
    scope: Literal['catalog','legacy_service_extension']
    values: list[float | None] = Field(max_length=MAX_ROWS)
    nonfinite: list[Literal[0,1,2,3]] = Field(max_length=MAX_ROWS,description='0 finite; 1 NaN; 2 +Infinity; 3 -Infinity')
    reason: list[Literal['legacy_nonfinite_unknown'] | None] = Field(max_length=MAX_ROWS,
        description='Unknown cause is explicit. No inferred unvoiced/failed label from a legacy NaN.')

    @field_validator('nonfinite',mode='before')
    @classmethod
    def integer_masks(cls,values):
        # Literal[0,1] otherwise normalizes False/True before the after validator.
        if isinstance(values,list) and any(type(v) is not int for v in values):
            raise ValueError('Mask must be an integer code')
        return values

    @model_validator(mode='after')
    def scientific_column(self):
        if len(self.values)!=len(self.nonfinite) or len(self.values)!=len(self.reason):raise ValueError('Column arrays must have equal lengths')
        if self.label!=PARAMETER_MAPPING.get(self.key,self.key) or self.unit!=parameter_unit(self.key):raise ValueError('Parameter label or unit mismatch')
        expected='catalog' if self.key in PARAMETER_MAPPING else 'legacy_service_extension'
        if self.scope!=expected:raise ValueError('Parameter scope mismatch')
        for value,mask,reason in zip(self.values,self.nonfinite,self.reason):
            if type(mask)!=int:raise ValueError('Mask must be an integer code')
            if mask==0:
                if value is None or reason is not None:raise ValueError('Finite sample requires value and null reason')
            elif value is not None or reason!='legacy_nonfinite_unknown':raise ValueError('Nonfinite sample requires null value and explicit unknown cause')
        return self


class AcousticTextColumn(WireModel):
    key: str = Field(pattern=r'^text_.+',max_length=180)
    values: list[Annotated[str,Field(max_length=32767)]] = Field(max_length=MAX_ROWS)


class AcousticResult(WireModel):
    schema_version: Literal['m01/1'] = SCHEMA_VERSION
    metadata: AcousticMetadata
    times_s: list[FiniteTime] = Field(max_length=MAX_ROWS)
    numeric: list[AcousticNumericColumn] = Field(max_length=82)
    text: list[AcousticTextColumn] = Field(max_length=64)
    column_order: list[Annotated[str,Field(max_length=180)]] = Field(min_length=1,max_length=147)

    @model_validator(mode='after')
    def consistent_result(self):
        n=len(self.times_s);columns=[*self.numeric,*self.text]
        if (n+1)*(1+len(columns))>MAX_CELLS:raise ValueError('Result cell budget exceeded')
        if any(a>=b for a,b in zip(self.times_s,self.times_s[1:])):raise ValueError('Times must increase strictly')
        shift=self.metadata.config.settings.frameshift_ms
        if any(not math.isclose(t,i*shift/1000.,rel_tol=1e-12,abs_tol=1e-12) for i,t in enumerate(self.times_s)):
            raise ValueError('Times disagree with recorded frame grid')
        if self.times_s and self.times_s[-1]>=self.metadata.decoded.sample_count/self.metadata.decoded.sample_rate_hz:raise ValueError('Frame time exceeds audio')
        if any(len(c.values)!=n for c in columns):raise ValueError('Result columns must match time axis')
        keys=['Time_s',*(c.key for c in columns)]
        if len(set(keys))!=len(keys) or self.column_order[0]!='Time_s' or len(self.column_order)!=len(keys) or set(self.column_order)!=set(keys):raise ValueError('Result column order mismatch')
        if self.metadata.config.selection.mode=='catalog' and any(c.key not in self.metadata.config.selection.keys for c in self.numeric):raise ValueError('Unselected numeric output')
        if sum(len(s.encode('utf-8')) for c in self.text for s in c.values)>2_000_000:raise ValueError('Annotation text budget exceeded')
        return self


class AcousticResultFile(WireModel):
    asset_id: Identifier
    format: Literal['xlsx','sqlite']
    size_bytes: int = Field(gt=0,le=LEGACY_QUOTA_BYTES)
    sha256: Hash
    expires_at: FiniteTime | None


class AcousticFileManifest(WireModel):
    # Historical manifests omit the field. New publishers must set version 2.
    policy_version: PolicyVersion = LEGACY_POLICY_VERSION
    kind: Literal['acoustic_file'] = 'acoustic_file'
    schema_version: Literal['m01/1'] = SCHEMA_VERSION
    complete: Literal[True] = True
    job_id: Identifier
    metadata: AcousticMetadata
    completed_at: FiniteTime
    retention: Literal['server','local']
    expires_at: FiniteTime | None
    row_count: int = Field(gt=0,le=MAX_ROWS)
    files: list[AcousticResultFile] = Field(min_length=2,max_length=2)

    @model_validator(mode='after')
    def complete_pair(self):
        if {f.format for f in self.files}!={'xlsx','sqlite'} or len({f.asset_id for f in self.files})!=2:raise ValueError('Complete file requires distinct XLSX and SQLite')
        if any(f.asset_id in {i.asset_id for i in self.metadata.inputs} for f in self.files):raise ValueError('Outputs cannot replace inputs')
        if any(f.expires_at!=self.expires_at for f in self.files):raise ValueError('Paired output lifetime mismatch')
        if self.retention=='local':
            if self.expires_at is not None:raise ValueError('Local result has no server expiry')
        else:
            if self.expires_at is None or not self.completed_at<self.expires_at<=self.completed_at+retention_seconds(self.policy_version):raise ValueError('Server result exceeds its storage policy retention limit')
            if self.policy_version == POLICY_VERSION:
                if sum(f.size_bytes for f in self.files)>QUOTA_BYTES:raise ValueError('Server result exceeds storage quota')
            if any(i.expires_at is None or i.expires_at<=self.completed_at for i in self.metadata.inputs):raise ValueError('Input was not live at completion')
        return self


BatchState = Literal['not_started','queued','running','cancel_requested','succeeded','failed','cancelled','interrupted']


class AcousticBatchItem(WireModel):
    progress: float = Field(default=0.,ge=0,le=1)
    index: int = Field(ge=0,le=9999)
    audio_asset_id: Identifier
    job_id: Identifier | None
    state: BatchState
    error_code: Code | None = None

    @model_validator(mode='after')
    def job_presence(self):
        if (self.state=='not_started')!=(self.job_id is None):raise ValueError('Only not_started has no child job')
        if self.state in ('succeeded','not_started') and self.error_code is not None:raise ValueError('Unexpected error code')
        return self


class AcousticBatchCounts(WireModel):
    not_started: int = Field(ge=0,le=10000)
    queued: int = Field(ge=0,le=10000)
    running: int = Field(ge=0,le=10000)
    cancel_requested: int = Field(ge=0,le=10000)
    succeeded: int = Field(ge=0,le=10000)
    failed: int = Field(ge=0,le=10000)
    cancelled: int = Field(ge=0,le=10000)
    interrupted: int = Field(ge=0,le=10000)


class AcousticBatchSummary(WireModel):
    kind: Literal['acoustic_batch'] = 'acoustic_batch'
    schema_version: Literal['m01/1'] = SCHEMA_VERSION
    batch_id: Identifier
    total: int = Field(ge=1,le=10000)
    closed: bool
    complete: bool = Field(description='True only when the closed batch succeeded for every requested file')
    counts: AcousticBatchCounts
    items: list[AcousticBatchItem] = Field(min_length=1,max_length=10000)

    @model_validator(mode='after')
    def truthful_summary(self):
        if self.total!=len(self.items) or [i.index for i in self.items]!=list(range(self.total)):raise ValueError('Batch items must cover the whole ordered list')
        ids=[i.job_id for i in self.items if i.job_id is not None]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate child job')
        for state,count in self.counts.model_dump().items():
            if sum(i.state==state for i in self.items)!=count:raise ValueError('Batch count mismatch')
        if self.closed and any(i.state in ('queued','running','cancel_requested') for i in self.items):raise ValueError('Closed batch still has active jobs')
        if self.complete!=(self.closed and self.counts.succeeded==self.total):raise ValueError('Batch complete must mean all files succeeded')
        return self


ACOUSTIC_SCHEMAS=(AcousticRequest,AcousticResult,AcousticFileManifest,AcousticBatchSummary)
