"""Owned M03 compatibility child: WAV bytes -> complete, hashed export bundle."""
import hashlib
import io
import json
import struct
import sys

INPUT_BYTES = 64_000_000
MAX_SAMPLES = 5_760_000
MAX_SECONDS = 120
MAX_INVERSE_SAMPLES = 960_000


def digest(raw): return hashlib.sha256(raw).hexdigest()


def prepare(raw, config, input_name='egg.wav', *, reaper=None, recording=None):
    from .egg_runtime import fingerprint
    runtime = fingerprint()
    import numpy as np
    from scipy.io import wavfile
    from phonetic_core.egg import EGGConfig, prepare as load, analyze_events
    from phonetic_core.egg.f0 import glottal_movement
    from .egg_f0 import populate, evidence
    from phonetic_core.egg.inverse import inverse_filter
    from ptb_api.egg_models import EggTaskConfig, expected_names
    from .acoustic_errors import AcousticFailure
    settings = EggTaskConfig.model_validate(config)
    if recording:
        from phonetic_core.egg.bounded import TimeView
        fs,samples,input_hash=recording.fs,TimeView(recording.frames,recording.fs),recording.sha
    else:
        if not 0 < len(raw) <= INPUT_BYTES: raise AcousticFailure('egg_input_budget')
        try: fs, samples = wavfile.read(io.BytesIO(raw))
        except (ValueError, OSError, EOFError): raise AcousticFailure('invalid_audio') from None
        if samples.ndim != 2 or samples.shape[1] != 2 or not len(samples): raise AcousticFailure('egg_stereo_required')
        if not 8000 <= fs <= 96000: raise AcousticFailure('egg_sample_rate')
        if len(samples) > MAX_SAMPLES or len(samples)/fs > MAX_SECONDS: raise AcousticFailure('egg_input_budget')
        input_hash=digest(raw)
    first = int(settings.roi_start*fs)
    last = len(samples) if settings.roi_end is None else int(settings.roi_end*fs)
    if not 0 <= first < last <= len(samples): raise AcousticFailure('egg_invalid_roi')
    if settings.mode == 'inverse' and last-first > MAX_INVERSE_SAMPLES: raise AcousticFailure('egg_inverse_budget')
    font_evidence=None
    if settings.font is not None and (settings.mode=='single' or settings.mode=='batch' and settings.generate_images):
        from .fonts import resolve_fonts
        font_evidence=resolve_fonts(settings.font)
    numerical = EGGConfig.for_workbench(**{k:v for k,v in settings.model_dump().items() if k in EGGConfig.__dataclass_fields__})
    result = recording.result(numerical,settings.flip_channels) if recording else load(samples,int(fs),numerical,flip_channels=settings.flip_channels)
    # CQ/SQ and micro markers run their own V2 local-window detection below.
    # Full-file events are only consumed by GCI F0 or export/inverse modes.
    # Preserve full-file normalization/detrending/filtering for every preview.
    if settings.mode != 'preview' or settings.keep_gci_f0:
        if recording and settings.mode=='inverse':
            from phonetic_core.egg.bounded import analyze_events as bounded_events
            result=bounded_events(result,numerical,span=(first,last))
        else: result = analyze_events(result, numerical)
    first_time, last_time = first/fs, last/fs
    blobs = {}; masked = 0; preview=None; inverse_view=None
    if settings.mode == 'inverse':
        audio = result.audio_signal[first:last]
        gcis = np.asarray(result.gci_times); gcis = gcis[(gcis >= first_time) & (gcis < last_time)]-first_time
        filtered = inverse_filter(audio, int(fs), gcis, lp_order=settings.lp_order)
        from phonetic_core.egg.preview import inverse_comparison
        from ptb_api.egg_models import EggInverseData
        frequencies, spectra, times, waves = inverse_comparison(audio,filtered,result.egg_signal_processed[first:last],int(fs))
        if len(frequencies)>30000:
            stride=int(np.ceil(len(frequencies)/30000))
            frequencies=frequencies[::stride];spectra=[v[::stride] for v in spectra]
        inverse_view = EggInverseData(frequencies_hz=frequencies.tolist(),audio_db=spectra[0].tolist(),
            inverse_db=spectra[1].tolist(),egg_db=spectra[2].tolist(),relative_times_s=times.tolist(),
            audio_values=waves[0].tolist(),inverse_values=waves[1].tolist(),egg_values=waves[2].tolist(),
            full_egg_values=result.egg_signal_processed[first:last].tolist(),sample_rate_hz=int(fs),
            lp_order=settings.lp_order or int(fs/1000)+6,gci_count=len(gcis),
            fixed_window_crossings=int(np.count_nonzero(np.diff((gcis*fs).astype(int)) < 1+int(.003*fs)))).model_dump()
        for name, values in [('egg_ORIG.wav',audio),('egg_IF.wav',filtered)]:
            stream = io.BytesIO(); wavfile.write(stream,int(fs),values.astype(np.float64)); blobs[name] = stream.getvalue()
    else:
        populate(result,settings,reaper,praat=settings.mode == 'single' or settings.keep_praat_f0 or settings.glottal_movement)
        if settings.glottal_movement:
            result.glottal_movement_events = glottal_movement(result.audio_f0_times,result.audio_f0_values)
        if settings.mode == 'preview':
            from .egg_preview import preview_files
            preview, blobs = preview_files(result,numerical,settings,first,last)
        else:
            from .egg_exports import csv_bytes, plot_bytes
            blobs['egg_DATA.csv'], masked = csv_bytes(result,numerical,settings,first_time,last_time)
            if settings.mode == 'single' or settings.generate_images:
                blobs.update(plot_bytes(result,numerical,settings,first,last,font_evidence))
    metadata = dict(schema_version='m03/1',method_version=result.method_version,export_policy=settings.export_policy,
        config=settings.model_dump(),render_fonts=font_evidence,input_sha256=input_hash,sample_rate_hz=int(fs),sample_count=len(samples),
        selection=dict(start_sample=first,end_sample=last,start_s=first_time,end_s=last_time,interval='half-open'),
        channel_roles=['audio','egg'] if settings.flip_channels else ['egg','audio'],
        normalized_peak=.7,source_ids=list(result.source_ids),runtime=runtime,
        csv_grid='gci-interpolated' if settings.mode=='batch' else 'outer-join-native-grids',
        csv_mask='20ms-mean-absolute-audio-below-threshold' if settings.mode=='batch' else 'none',
        csv_masked_rows=masked,plots_mask='none',praat_time='pitch.xs()',
        local_cq_policy='legacy-100ms-padding-repeat-filter',waveform_time='sample-index/fs',
        inverse=(dict(sample_count=last-first,lp_order=settings.lp_order or int(fs/1000)+6,
            autocorrelation_policy='legacy-through-48000-samples;requested-lags-for-larger-roi',
            original='normalized-analysis-audio',estimate='simplified-closed-phase-inverse-filter',wav_subtype='FLOAT64') if settings.mode=='inverse' else None))
    metadata['f0_analysis']=evidence(settings,reaper if settings.mode!='inverse' else None)
    if recording:
        metadata['bounded']=dict(revision='egg-bounded/2',block_seconds=20,filter_halo='pole-decay-1e-10; minimum-1s; maximum-30s',filter_form='butterworth4-SOS-forward-backward',local_filter_form='butterworth4-SOS-forward-backward; legacy-padding-and-repeat-filter-retained',
            f0_path='per-block-20s-1s-context',gci_f0_outlier_reference='complete-event-grid',
            interactive_gci_f0_reference='viewport-event-grid',waveform_display='4096-bins-min-max',spectral_display='2048-columns-max-PSD',
            numerical_identity_to_whole_file=False)
    if settings.keep_reaper_f0 and settings.mode!='inverse':metadata['source_ids'].append('SRC-REAPER')
    if preview is not None:
        metadata.update(preview=preview,csv_grid=None,csv_mask=None)
    if inverse_view is not None: metadata['inverse_view']=inverse_view
    names = expected_names(settings)
    from .egg_export_names import export_names
    metadata.update(input_name=input_name,export_names=export_names(input_name,settings.mode,first_time,last_time,names))
    blobs['egg.ptb.json'] = json.dumps(metadata,ensure_ascii=False,allow_nan=False).encode()
    if set(blobs) != set(names): raise AcousticFailure('egg_incomplete_export')
    header = dict(kind='prepared_egg',audio_sha256=input_hash,files=[dict(name=n,format=n.rsplit('.',1)[1],
        size_bytes=len(blobs[n]),sha256=digest(blobs[n])) for n in names])
    encoded = json.dumps(header).encode(); payload = struct.pack('<Q',len(encoded))+encoded+b''.join(blobs[n] for n in names)
    if len(payload)>(256_000_000 if recording else 64_000_000): raise AcousticFailure('analysis_output_limit')
    return payload


def main():
    with open(sys.argv[1],'rb') as handle:
        header = json.loads(handle.readline(16384)); raw = handle.read(INPUT_BYTES+1)
    if header.get('source_path'):
        from .egg_recording import Recording
        source=Recording(header['source_path'])
        try:
            if source.sha!=header['sha256']:raise ValueError('Input changed')
            from .egg_f0 import native_engine
            with native_engine(header) as reaper:
                payload=prepare(b'',header['config'],header.get('input_name','egg.wav'),reaper=reaper,recording=source) if source.long else prepare(source.path.read_bytes(),header['config'],header.get('input_name','egg.wav'),reaper=reaper)
            with open(sys.argv[2],'wb',buffering=0) as out:
                for offset in range(0,len(payload),65536):out.write(payload[offset:offset+65536])
            return
        finally:source.close()
    if len(raw)>INPUT_BYTES or digest(raw)!=header['sha256']: raise ValueError('Input changed')
    from .egg_f0 import native_engine
    with native_engine(header) as reaper:
        payload = prepare(raw,header['config'],header.get('input_name','egg.wav'),reaper=reaper)
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
