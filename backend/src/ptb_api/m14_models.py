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
    action:Literal['inspect','preview','export']
    computation_revision:Literal['m14/1','m14/2']='m14/1'
    character_column:int=Field(default=1,ge=1,le=64)
    ipa_column:int=Field(default=2,ge=1,le=64)
    note_column:int|None=Field(default=3,ge=1,le=64)
    start_row:int=Field(default=2,ge=1,le=10001)
    table_index:int=Field(default=0,ge=0,le=10000)
    encoding:Literal['auto','utf-8-sig','gb18030','utf-16']='auto'
    delimiter:Literal['auto','tab','comma','semicolon','chinese_comma','space']='auto'
    skip_first_row:bool=True
    consonant_only_as_zero_initial:bool=True
    settings:M14Settings|None=None
    font:FigureFontSnapshot|None=None

    @model_validator(mode='after')
    def export_fields(self):
        if self.action!='inspect' and self.computation_revision=='m14/2' and len({self.character_column,self.ipa_column})!=2:
            raise ValueError('m14_column_selection')
        if self.action!='inspect' and self.computation_revision=='m14/2' and self.note_column in (self.character_column,self.ipa_column):
            raise ValueError('m14_column_selection')
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
