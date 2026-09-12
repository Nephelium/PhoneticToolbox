"""Owned M03 compatibility child: WAV bytes -> complete, hashed export bundle."""
import hashlib
import io
import json
import struct
import sys

INPUT_BYTES = 64_000_000
MAX_SAMPLES = 2_880_000
MAX_SECONDS = 60
MAX_INVERSE_SAMPLES = 48_000


def digest(raw): return hashlib.sha256(raw).hexdigest()


def prepare(raw, config):
    from .egg_runtime import fingerprint
    runtime = fingerprint()
    import numpy as np
    from scipy.io import wavfile
    from phonetic_core.egg import EGGConfig, prepare as load, analyze_events
    from phonetic_core.egg.f0 import praat_pitch, glottal_movement
    from phonetic_core.egg.inverse import inverse_filter
    from ptb_api.egg_models import EggTaskConfig, expected_names
    from .acoustic_errors import AcousticFailure
    from .egg_exports import csv_bytes, plot_bytes
    settings = EggTaskConfig.model_validate(config)
    if not 0 < len(raw) <= INPUT_BYTES: raise AcousticFailure('egg_input_budget')
    try: fs, samples = wavfile.read(io.BytesIO(raw))
    except (ValueError, OSError, EOFError): raise AcousticFailure('invalid_audio') from None
    if samples.ndim != 2 or samples.shape[1] != 2 or not len(samples): raise AcousticFailure('egg_stereo_required')
    if not 8000 <= fs <= 96000: raise AcousticFailure('egg_sample_rate')
    if len(samples) > MAX_SAMPLES or len(samples)/fs > MAX_SECONDS: raise AcousticFailure('egg_input_budget')
    first = int(settings.roi_start*fs)
    last = len(samples) if settings.roi_end is None else int(settings.roi_end*fs)
    if not 0 <= first < last <= len(samples): raise AcousticFailure('egg_invalid_roi')
    if settings.mode == 'inverse' and last-first > MAX_INVERSE_SAMPLES: raise AcousticFailure('egg_inverse_budget')
    numerical = EGGConfig.for_workbench(**{k:v for k,v in settings.model_dump().items() if k in EGGConfig.__dataclass_fields__})
    result = analyze_events(load(samples, int(fs), numerical, flip_channels=settings.flip_channels), numerical)
    first_time, last_time = first/fs, last/fs
    blobs = {}; masked = 0; font_evidence=None
    if settings.font is not None and (settings.mode=='single' or settings.generate_images):
        from .fonts import resolve_fonts
        font_evidence=resolve_fonts(settings.font)
    if settings.mode == 'inverse':
        audio = result.audio_signal[first:last]
        gcis = np.asarray(result.gci_times); gcis = gcis[(gcis >= first_time) & (gcis < last_time)]-first_time
        filtered = inverse_filter(audio, int(fs), gcis, lp_order=settings.lp_order)
        for name, values in [('egg_ORIG.wav',audio),('egg_IF.wav',filtered)]:
            stream = io.BytesIO(); wavfile.write(stream,int(fs),values.astype(np.float64)); blobs[name] = stream.getvalue()
    else:
        if settings.keep_praat_f0 or settings.glottal_movement:
            track = praat_pitch(result.audio_signal,int(fs))
            result.audio_f0_times, result.audio_f0_values = track.times,track.values
        if settings.glottal_movement:
            result.glottal_movement_events = glottal_movement(result.audio_f0_times,result.audio_f0_values)
        blobs['egg_DATA.csv'], masked = csv_bytes(result,numerical,settings,first_time,last_time)
        if settings.mode == 'single' or settings.generate_images:
            blobs.update(plot_bytes(result,numerical,settings,first,last,font_evidence))
    metadata = dict(schema_version='m03/1',method_version=result.method_version,export_policy=settings.export_policy,
        config=settings.model_dump(),render_fonts=font_evidence,input_sha256=digest(raw),sample_rate_hz=int(fs),sample_count=len(samples),
        selection=dict(start_sample=first,end_sample=last,start_s=first_time,end_s=last_time,interval='half-open'),
        channel_roles=['audio','egg'] if settings.flip_channels else ['egg','audio'],
        normalized_peak=.7,source_ids=list(result.source_ids),runtime=runtime,
        csv_grid='gci-interpolated' if settings.mode=='batch' else 'outer-join-native-grids',
        csv_mask='20ms-mean-absolute-audio-below-threshold' if settings.mode=='batch' else 'none',
        csv_masked_rows=masked,plots_mask='none',praat_time='pitch.xs()',
        local_cq_policy='legacy-100ms-padding-repeat-filter',waveform_time='sample-index/fs',
        inverse=(dict(sample_count=last-first,lp_order=settings.lp_order or int(fs/1000)+6,
            original='normalized-analysis-audio',estimate='simplified-closed-phase-inverse-filter',wav_subtype='FLOAT64') if settings.mode=='inverse' else None))
    blobs['egg.ptb.json'] = json.dumps(metadata,ensure_ascii=False,allow_nan=False).encode()
    names = expected_names(settings)
    if set(blobs) != set(names): raise AcousticFailure('egg_incomplete_export')
    header = dict(kind='prepared_egg',audio_sha256=digest(raw),files=[dict(name=n,format=n.rsplit('.',1)[1],
        size_bytes=len(blobs[n]),sha256=digest(blobs[n])) for n in names])
    encoded = json.dumps(header).encode(); payload = struct.pack('<Q',len(encoded))+encoded+b''.join(blobs[n] for n in names)
    if len(payload)>64_000_000: raise AcousticFailure('analysis_output_limit')
    return payload


def main():
    with open(sys.argv[1],'rb') as handle:
        header = json.loads(handle.readline(16384)); raw = handle.read(INPUT_BYTES+1)
    if len(raw)>INPUT_BYTES or digest(raw)!=header['sha256']: raise ValueError('Input changed')
    payload = prepare(raw,header['config'])
    with open(sys.argv[2],'wb',buffering=0) as out:
        for offset in range(0,len(payload),65536): out.write(payload[offset:offset+65536])


def run():
    try: main()
    except Exception as exc:
        from .acoustic_errors import public_error
        code = getattr(exc,'code',None)
        known = {'filter_failed':'egg_filter_failed','filter_cutoff':'egg_filter_failed',
                 'invalid_filter_cutoff':'egg_filter_failed','inverse_unavailable':'egg_inverse_unavailable',
                 'pitch_failed':'egg_pitch_failed','invalid_audio':'invalid_audio',
                 'invalid_signal':'invalid_audio','signal_float32_overflow':'invalid_audio',
                 'invalid_roi':'egg_invalid_roi','requires_two_channels':'egg_stereo_required'}
        header = json.dumps({'error':known.get(code,public_error(exc))}).encode()
        try:
            with open(sys.argv[2],'wb',buffering=0) as out: out.write(struct.pack('<Q',len(header))+header)
        except Exception: raise SystemExit(2) from None


if __name__=='__main__': run()
