"""Explicit opt-in M01 bounded computation; old requests serialize unchanged."""
from typing import Literal
from pydantic import Field, model_validator
from .models import WireModel,Identifier

class JointEggSettings(WireModel):
    egg_channel:int=Field(default=0,ge=0,le=7)
    storage:Literal['aligned','cycles']='aligned'
    smooth_ms:float=Field(default=20.,ge=0,le=1000,allow_inf_nan=False)
    max_gap_ms:float=Field(default=50.,gt=0,le=1000,allow_inf_nan=False)
    derived:bool=False
    highpass_cutoff:float=Field(default=25.,gt=0,le=10000,allow_inf_nan=False)
    lowpass_cutoff:float=Field(default=2000.,gt=0,le=48000,allow_inf_nan=False)
    gci_method:Literal['slope','scale']='slope'
    goi_method:Literal['slope','scale']='scale'
    auto_prominence:bool=True
    peak_prominence:float=Field(default=.01,ge=0,le=1,allow_inf_nan=False)
    valley_prominence:float=Field(default=.01,ge=0,le=1,allow_inf_nan=False)
    silence_threshold:float=Field(default=.01,ge=0,le=1,allow_inf_nan=False)
    @model_validator(mode='after')
    def filters(self):
        if self.highpass_cutoff>=self.lowpass_cutoff:raise ValueError('Highpass must be below lowpass')
        return self

class AcousticExtended(WireModel):
    revision:Literal['bounded/1']='bounded/1'
    max_duration_s:Literal[1800]=1800
    audio_channel:int|None=Field(default=None,ge=0,le=7)
    egg:JointEggSettings|None=None
    channel_overrides:dict[str,int]=Field(default_factory=dict,max_length=1000)
    @model_validator(mode='after')
    def channels(self):
        if self.egg and (self.audio_channel is None or self.audio_channel==self.egg.egg_channel):
            raise ValueError('Audio and EGG require distinct channels')
        if any(type(v)!=int or not 0<=v<=7 or len(k)>220 for k,v in self.channel_overrides.items()):
            raise ValueError('Invalid channel override')
        return self

class ParameterView(WireModel):
    start:float=Field(default=0,ge=0,le=1800,allow_inf_nan=False)
    end:float=Field(gt=0,le=1800,allow_inf_nan=False)
    width:int=Field(default=1200,ge=32,le=4000)
    parameters:list[str]=Field(default_factory=list,max_length=128)

class ParameterWindowRequest(WireModel):
    asset_id:Identifier
    view:ParameterView|None=None
