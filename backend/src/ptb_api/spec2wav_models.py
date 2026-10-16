"""Versioned image reconstruction request; no new persistence schema."""
from typing import Literal
from pydantic import Field,model_validator
from .models import WireModel,Identifier,IdempotencyKey
from .acoustic_models import AcousticAssetRef
from .acoustic_batch_models import AcousticManagedFile


class ImagePoint(WireModel):
    x:float=Field(ge=0,le=1)
    y:float=Field(ge=0,le=1)


class SpectralStroke(WireModel):
    color:Literal[0,255]=0
    size:int=Field(default=12,ge=1,le=200)
    size_unit:Literal['legacy','px']='legacy'
    opacity:float=Field(default=1,ge=0,le=1)
    points:list[ImagePoint]=Field(min_length=1,max_length=4096)


class Spec2WavConfig(WireModel):
    mode:Literal['image','image_draw','audio_draw']='image'
    invert_image:bool=False
    vocoder:Literal['none','bigvgan']='none'
    algorithm:Literal['griffin_lim','neural_pipeline']='griffin_lim'
    reconstruction_profile:Literal['legacy','calibrated']='legacy'
    inference_device:Literal['auto','cpu','cuda']='auto'
    phase_initialization:Literal['auto','pulse','random']='auto'
    frequency_axis:Literal['linear','log']='linear'
    auto_parameters:bool=False
    strokes:list[SpectralStroke]=Field(default_factory=list,max_length=256)
    channel:int=Field(default=0,ge=0,le=1)
    audio_fft:Literal[512,1024,2048]=1024
    dynamic_range:float=Field(default=60,ge=20,le=120)
    time_start:float=Field(default=0,ge=0,le=86400)
    time_end:float=Field(default=1,gt=0,le=86430)
    freq_start:float=Field(default=0,ge=0,lt=48000)
    freq_end:float=Field(default=11025,ge=1,le=48000)
    min_db:float=Field(default=-30,ge=-160,le=20)
    max_db:float=Field(default=0,ge=-160,le=20)
    win_length_ms:float=Field(default=10,gt=0,le=1000)
    n_iter:int=Field(default=32,ge=1,le=256)
    target_sr:Literal[0,8000,16000,22050,24000,32000,44100,48000,96000]=44100
    seed:int=Field(default=0,ge=0,le=4294967295)
    corners:list[ImagePoint] | None=Field(default=None,min_length=4,max_length=4)

    @model_validator(mode='before')
    @classmethod
    def algorithm_defaults(cls,value):
        # Existing GL snapshots omitted this flag and remain manual. Earlier
        # neural snapshots also omitted it, with automatic prediction enabled.
        if isinstance(value,dict) and value.get('algorithm')=='neural_pipeline' and 'auto_parameters' not in value:
            return dict(value,auto_parameters=True)
        return value

    @model_validator(mode='after')
    def ranges(self):
        if not (0<self.time_end-self.time_start<=30 and self.freq_start<self.freq_end and self.min_db<self.max_db):raise ValueError('Invalid calibration range')
        if sum(len(s.points) for s in self.strokes)>8192:raise ValueError('Stroke budget exceeded')
        if self.mode=='image' and self.strokes:raise ValueError('Image mode does not accept strokes')
        if self.mode=='audio_draw' and self.corners:raise ValueError('Audio does not accept image calibration')
        if self.mode=='audio_draw' and self.invert_image:raise ValueError('Audio does not accept image inversion')
        if self.mode=='audio_draw' and self.vocoder!='none':raise ValueError('Audio editing preserves original phase')
        if self.mode=='audio_draw' and (self.algorithm!='griffin_lim' or self.frequency_axis!='linear'):raise ValueError('Audio editing preserves original phase')
        if self.frequency_axis=='log' and self.freq_start<=0:raise ValueError('Log frequency requires a positive lower bound')
        if self.algorithm=='neural_pipeline' and not self.auto_parameters and self.win_length_ms>160:raise ValueError('Neural pipeline window exceeds 160 ms')
        if self.reconstruction_profile=='calibrated' and not self.auto_parameters and self.win_length_ms>160:raise ValueError('Calibrated window exceeds 160 ms')
        if self.inference_device=='cuda' and self.algorithm!='neural_pipeline' and self.vocoder=='none':raise ValueError('GPU requires a neural model')
        return self

    def snapshot(self):
        # Preserve existing request hashes and retries when inversion is disabled.
        excluded={'mode','strokes','channel','audio_fft','dynamic_range'} if self.mode=='image' else set()
        if not self.invert_image:excluded.add('invert_image')
        if self.vocoder=='none':excluded.add('vocoder')
        if self.algorithm=='griffin_lim':excluded.add('algorithm')
        if self.reconstruction_profile=='legacy':excluded.add('reconstruction_profile')
        if self.inference_device=='auto':excluded.add('inference_device')
        if self.phase_initialization=='auto':excluded.add('phase_initialization')
        if self.frequency_axis=='linear':excluded.add('frequency_axis')
        if self.auto_parameters==(self.algorithm=='neural_pipeline'):excluded.add('auto_parameters')
        result=self.model_dump(exclude=excluded)
        for stroke in result.get('strokes',[]):
            if stroke.get('size_unit')=='legacy':stroke.pop('size_unit',None)
        return result


class Spec2WavRequest(WireModel):
    schema_version:Literal['m09/1']='m09/1'
    project_id:Identifier
    idempotency_key:IdempotencyKey
    image:AcousticAssetRef
    config:Spec2WavConfig


class Spec2WavPreview(WireModel):
    image_base64:str
    width:int
    height:int
    source_sha256:str
    sample_rate:int|None=None
    samples:int|None=None
    channels:int|None=None
    channel:int|None=None
    duration:float|None=None
    n_fft:int|None=None
    hop_length:int|None=None
    dynamic_range:float|None=None
    reference_amplitude:float|None=None
    prediction:dict|None=None


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
