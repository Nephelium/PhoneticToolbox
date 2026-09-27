"""M03/1: immutable per-file task, shared P06/P07 persistence (no DDL)."""
from typing import Literal, Annotated
from pydantic import Field, model_validator
from .models import WireModel, Identifier, IdempotencyKey
from .acoustic_models import AcousticAssetRef
from .acoustic_batch_models import AcousticManagedFile
from .font_models import FigureFontSnapshot


class EggTaskConfig(WireModel):
    font: FigureFontSnapshot | None = None  # None preserves historical task rendering.
    mode: Literal['single', 'batch', 'inverse', 'preview'] = 'single'
    micro_center: float | None = Field(default=None, ge=0, le=120)
    micro_width_ms: float = Field(default=50., ge=5, le=5000)
    flip_channels: bool = False
    signal_mode: Literal['raw', 'filtered'] = 'filtered'
    roi_start: float = Field(default=0.0, ge=0, le=86400)
    roi_end: float | None = Field(default=None, gt=0, le=86400)
    gci_method: Literal['slope', 'scale'] = 'slope'
    goi_method: Literal['slope', 'scale'] = 'scale'
    peak_prominence: float = Field(default=.01, ge=0, le=10)
    valley_prominence: float = Field(default=.01, ge=0, le=10)
    auto_prominence: bool = True
    highpass_cutoff: float = Field(default=25.0, gt=0, lt=48000)
    lowpass_cutoff: float = Field(default=1000.0, gt=0, lt=48000)
    spec_window_ms: float = Field(default=20.0, ge=5, le=50)
    spec_vmin: float = Field(default=-70.0, ge=-160, le=20)
    spec_vmax: float = Field(default=-10.0, ge=-160, le=20)
    keep_praat_f0: bool = True
    keep_gci_f0: bool = True
    glottal_movement: bool = False
    silence_threshold: float = Field(default=.01, ge=0, le=1)
    generate_images: bool = False
    lp_order: int | None = Field(default=None, ge=1, le=256)
    export_policy: Literal['sample-aligned/1'] = 'sample-aligned/1'

    @model_validator(mode='after')
    def ranges(self):
        if self.mode != 'preview' and (self.micro_center is not None or self.micro_width_ms != 50.):
            raise ValueError('Micro viewport only applies to preview')
        if self.highpass_cutoff >= self.lowpass_cutoff or self.spec_vmin >= self.spec_vmax:
            raise ValueError('Invalid filter or spectral range')
        if (self.roi_end is None and self.roi_start != 0) or (self.roi_end is not None and self.roi_end <= self.roi_start):
            raise ValueError('Specify a nonempty ROI or the whole file')
        if self.mode == 'batch' and (self.roi_start != 0 or self.roi_end is not None or self.signal_mode != 'filtered'):
            raise ValueError('Batch analysis uses the complete filtered file')
        if self.mode == 'inverse' and (self.roi_end is None or self.roi_end-self.roi_start > 1):
            raise ValueError('Inverse filtering requires an explicit ROI of at most one second')
        if self.mode != 'inverse' and self.lp_order is not None:
            raise ValueError('LP order only applies to inverse filtering')
        return self


class EggRequest(WireModel):
    schema_version: Literal['m03/1'] = 'm03/1'
    project_id: Identifier
    idempotency_key: IdempotencyKey
    audio: AcousticAssetRef
    config: EggTaskConfig


def expected_names(config):
    names = ['egg.ptb.json']
    if config.mode == 'preview': return names + ['egg_AUDIO.wav', 'egg_PSD.png']
    if config.mode == 'inverse': return names + ['egg_ORIG.wav', 'egg_IF.wav']
    names += ['egg_DATA.csv']
    if config.mode == 'single' or config.generate_images:
        names += ['egg_CQ_SQ.png', 'egg_SPEC_F0.png', 'egg_WAVEFORMS.png']
    return names


from .storage_policy import PolicyVersion, LEGACY_POLICY_VERSION


class EggManifest(WireModel):
    policy_version: PolicyVersion = LEGACY_POLICY_VERSION
    kind: Literal['managed_egg_files'] = 'managed_egg_files'
    complete: Literal[True] = True
    operation: Literal['egg_analysis'] = 'egg_analysis'
    core_version: str
    files: list[AcousticManagedFile] = Field(min_length=2, max_length=5)

    @model_validator(mode='after')
    def complete_set(self):
        names = sorted(f.name for f in self.files)
        sets = [expected_names(EggTaskConfig()), expected_names(EggTaskConfig(mode='batch')),
                expected_names(EggTaskConfig(mode='inverse', roi_end=1.0)), expected_names(EggTaskConfig(mode='preview'))]
        if (names not in [sorted(s) for s in sets] or len({f.id for f in self.files}) != len(names)
                or sum(f.size_bytes for f in self.files) > 64_000_000):
            raise ValueError('Incomplete EGG export')
        return self


class EggSeries(WireModel):
    times: list[float] = Field(max_length=120000)
    values: list[float | None] = Field(max_length=120000)

    @model_validator(mode='after')
    def aligned(self):
        if len(self.times) != len(self.values): raise ValueError('Unaligned series')
        return self


class EggPreviewData(WireModel):
    schema_version: Literal['egg-preview/1'] = 'egg-preview/1'
    cq: EggSeries
    sq: EggSeries
    praat: EggSeries
    gci_f0: EggSeries
    audio: EggSeries
    egg: EggSeries
    gci: list[float] = Field(max_length=10000)
    goi: list[float] = Field(max_length=10000)
    movement: list[tuple[float, str]] = Field(max_length=1000)
    micro_center: float
    micro_width_ms: float
    micro_sample_stride: int = Field(default=1, ge=1, le=64)
    spectral_extent: tuple[float, float, float, float]
    spectral_shape: tuple[int, int]
    raster_shape: tuple[int, int]
    display_policy: Literal['legacy-roi-extent; grayscale-raster-only'] = 'legacy-roi-extent; grayscale-raster-only'
    micro_wave_policy: Literal['raw-100ms-padding-filter-crop'] = 'raw-100ms-padding-filter-crop'
    micro_event_policy: Literal['raw-50ms-padding'] = 'raw-50ms-padding'


DisplayValues = Annotated[list[float], Field(max_length=30000)]


class EggInverseData(WireModel):
    schema_version: Literal['egg-inverse-view/1'] = 'egg-inverse-view/1'
    frequencies_hz: DisplayValues
    audio_db: DisplayValues
    inverse_db: DisplayValues
    egg_db: DisplayValues
    relative_times_s: DisplayValues
    audio_values: DisplayValues
    inverse_values: DisplayValues
    egg_values: DisplayValues
    spectral_policy: Literal['pad-44100-periodic-hamming-fft-80db-floor'] = 'pad-44100-periodic-hamming-fft-80db-floor'
