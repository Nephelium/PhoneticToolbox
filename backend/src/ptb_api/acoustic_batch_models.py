"""Public M01 ordered requests and managed output references; no runtime imports."""
from typing import Literal
from pydantic import Field,model_validator
from .models import WireModel,Identifier,IdempotencyKey
from .acoustic_models import AcousticInputs,AcousticAssetRef,AcousticConfigSnapshot,AcousticBatchSummary


class BatchInputs(AcousticInputs):
    parent_result: AcousticAssetRef | None=None
    legacy_result: AcousticAssetRef | None=None

    @model_validator(mode='after')
    def one_parameter_source(self):
        if self.parent_result and self.legacy_result:raise ValueError('Choose one parameter source')
        return self


class BatchRequest(WireModel):
    schema_version: Literal['m01-batch/1']='m01-batch/1'
    operation: Literal['acoustic_analysis','textgrid_segment']
    project_id: Identifier
    idempotency_key: IdempotencyKey
    inputs: list[BatchInputs]=Field(min_length=1,max_length=1000)
    config: AcousticConfigSnapshot | None=None
    layer: str | None=Field(default=None,min_length=1,max_length=180)

    @model_validator(mode='after')
    def scope(self):
        if len({i.audio.asset_id for i in self.inputs})!=len(self.inputs):raise ValueError('Duplicate batch audio')
        for item in self.inputs:
            ids=[v.asset_id for v in (item.audio,item.textgrid,item.lip,item.parent_result,item.legacy_result) if v is not None]
            if len(ids)!=len(set(ids)):raise ValueError('Duplicate input roles')
        if self.operation=='acoustic_analysis':
            if self.config is None or self.layer is not None or any(i.parent_result or i.legacy_result for i in self.inputs):
                raise ValueError('Analysis requires explicit config, not cutting tier or parent result')
        elif self.config is not None or self.layer is None or not self.layer.strip() or any(i.textgrid is None for i in self.inputs):
            raise ValueError('Segmentation requires an explicit tier and TextGrid for every selected audio')
        return self


class BatchView(WireModel):
    id: Identifier
    project_id: Identifier
    operation: Literal['acoustic_analysis','textgrid_segment']
    created_at: float
    updated_at: float
    cancel_requested: bool
    request_sha256: str
    audio_names: list[str]=Field(min_length=1,max_length=1000)
    summary: AcousticBatchSummary


class BatchList(WireModel):
    batches: list[BatchView]=Field(max_length=100)


class AcousticManagedFile(WireModel):
    id: Identifier
    name: str=Field(max_length=255)
    kind: Literal['result']='result'
    size_bytes: int=Field(ge=1,le=64_000_000)
    sha256: str=Field(pattern=r'^[0-9a-f]{64}$')
    expires_at: float | None


class AcousticTaskManifest(WireModel):
    kind: Literal['managed_acoustic_files']='managed_acoustic_files'
    complete: Literal[True]=True
    operation: Literal['acoustic_analysis','textgrid_segment']
    core_version: str
    files: list[AcousticManagedFile]=Field(min_length=1,max_length=3001)

    @model_validator(mode='after')
    def output_set(self):
        if len({f.id for f in self.files})!=len(self.files):raise ValueError('Duplicate output')
        if self.operation=='acoustic_analysis' and sorted(f.name for f in self.files)!=['result.ptb.json','result.ptb.sqlite','result.xlsx']:
            raise ValueError('Analysis requires both parameter formats and its versioned result')
        if sum(f.size_bytes for f in self.files)>64_000_000:raise ValueError('Result budget exceeded')
        return self
