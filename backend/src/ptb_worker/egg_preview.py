"""Bounded display serialization; all scientific series come from installed core."""
import io
import numpy as np
from PIL import Image
from scipy.io import wavfile
from phonetic_core.egg import cq_segment, events_segment
from phonetic_core.egg.preview import micro_waveforms
from phonetic_core.egg.export_series import spectral_series
from ptb_api.egg_models import EggPreviewData, EggSeries
from .acoustic_errors import AcousticFailure


def preview_files(result, config, settings, first, last):
    start, end = first/result.fs, last/result.fs
    center = settings.micro_center if settings.micro_center is not None else (start+end)/2
    if not 0 <= center <= len(result.time_vector)/result.fs:
        raise AcousticFailure('egg_invalid_roi')
    raw = settings.signal_mode == 'raw'
    def series(times, values, crop=True):
        times = np.asarray([] if times is None else times)
        values = np.asarray([] if values is None else values)
        take = (times >= start) & (times < end) if crop else np.ones(len(times), dtype=bool)
        return EggSeries(times=times[take].tolist(), values=[float(v) if np.isfinite(v) else None for v in values[take]])
    t, cq, sq = cq_segment(result, start, end, config, use_raw_signal=raw)
    micro_t, audio, egg = micro_waveforms(result, config, center, settings.micro_width_ms, raw=raw)
    lo = max(0., center-settings.micro_width_ms/2000)
    hi = min(len(result.time_vector)/result.fs, center+settings.micro_width_ms/2000)
    gci, goi, _ = events_segment(result, lo, hi, config, use_raw_signal=raw)
    power, freq, _ = spectral_series(result.audio_signal[first:last], result.fs, config.spec_window_ms)
    cell = (freq[-1]-freq[0])/len(freq)
    rows = min(len(freq), int(np.ceil(5000/cell))+2)
    visible = power[:rows].copy(); del power
    shape = visible.shape
    with np.errstate(divide='ignore'): np.log10(visible, out=visible)
    visible *= 10
    gray = (255*(1-np.clip((visible-settings.spec_vmin)/(settings.spec_vmax-settings.spec_vmin),0,1))).astype(np.uint8)
    raster = Image.fromarray(np.flipud(gray))
    raster.thumbnail((1024,512), Image.Resampling.BILINEAR)
    stream = io.BytesIO(); raster.save(stream, format='PNG')
    wav = io.BytesIO(); wavfile.write(wav, result.fs, result.audio_signal.astype(np.float32))
    data = EggPreviewData(cq=series(t,cq), sq=series(t,sq),
        praat=series(result.audio_f0_times,result.audio_f0_values),
        gci_f0=series(result.gci_f0_times,result.gci_f0_values) if settings.keep_gci_f0 else series([],[]),
        audio=series(micro_t,audio,False), egg=series(micro_t,egg,False),
        gci=[v for v in gci if lo <= v < hi], goi=[v for v in goi if lo <= v < hi],
        movement=[(t,v) for t,v in result.glottal_movement_events if start <= t < end],
        micro_center=center, micro_width_ms=settings.micro_width_ms,
        spectral_extent=(start,end,0.,rows*cell), spectral_shape=shape,
        raster_shape=(raster.height,raster.width))
    return data.model_dump(), {'egg_AUDIO.wav':wav.getvalue(),'egg_PSD.png':stream.getvalue()}
