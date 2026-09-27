"""M14-owned strict contracts; no database schema change."""
from typing import Literal
from pydantic import Field,model_validator
from .models import WireModel,Identifier,IdempotencyKey
from .acoustic_models import AcousticAssetRef
from .font_models import FigureFontSnapshot
from .acoustic_batch_models import AcousticManagedFile


class M14Settings(WireModel):
    tone_map:dict[str,str]=Field(max_length=10000)
    tone_order:list[str]=Field(max_length=10000)
    initial_order:list[str]=Field(max_length=10000)
    final_order:list[str]=Field(max_length=10000)
    initial_map:dict[str,str]=Field(default_factory=dict,max_length=10000)
    final_map:dict[str,str]=Field(default_factory=dict,max_length=10000)


class M14Config(WireModel):
    action:Literal['preview','export']
    skip_first_row:bool=True
    consonant_only_as_zero_initial:bool=True
    settings:M14Settings|None=None
    font:FigureFontSnapshot|None=None

    @model_validator(mode='after')
    def export_fields(self):
        if self.action=='export' and (self.settings is None or self.font is None):raise ValueError('m14_export_config_required')
        return self


class M14Request(WireModel):
    schema_version:Literal['m14/1']='m14/1'
    project_id:Identifier
    idempotency_key:IdempotencyKey
    table:AcousticAssetRef
    config:M14Config


from .storage_policy import PolicyVersion, LEGACY_POLICY_VERSION


class M14Manifest(WireModel):
    policy_version: PolicyVersion = LEGACY_POLICY_VERSION
    kind:Literal['managed_m14_files']='managed_m14_files'
    complete:Literal[True]=True
    operation:Literal['phonology_induction']='phonology_induction'
    core_version:str
    files:list[AcousticManagedFile]=Field(min_length=1,max_length=3)

    @model_validator(mode='after')
    def complete_set(self):
        from phonetic_core.transcription.phonology.models import NAMES
        names=[f.name for f in self.files]
        if names!=['m14-preview.json'] and sorted(names)!=sorted(NAMES):
            raise ValueError('m14_incomplete_output')
        return self
