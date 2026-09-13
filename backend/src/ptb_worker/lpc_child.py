"""Bounded WAV/TextGrid decoding -> pure LPC core -> complete in-memory bundle."""
import hashlib
import io
import json
import struct
import sys

INPUT_BYTES=64_000_000
GRID_BYTES=2_000_000
OUTPUT_BYTES=8_000_000


def digest(raw):return hashlib.sha256(raw).hexdigest()


def prepare(raw, config, input_name='audio.wav', textgrid=None):
    from .lpc_runtime import fingerprint
    runtime=fingerprint()
    import numpy as np
    from scipy.io import wavfile
    from phonetic_core.lpc import LPCConfig, MAX_ROI_SAMPLES, compute_spectrum, mono_samples, extract_label
    from ptb_api.lpc_models import LpcTaskConfig, LpcSpectrumData, LPC_NAMES
    from .acoustic_errors import AcousticFailure
    from .fonts import resolve_fonts
    from .lpc_exports import plot_bytes, export_names
    settings=LpcTaskConfig.model_validate(config)
    if not 0<len(raw)<=INPUT_BYTES:raise AcousticFailure('lpc_input_budget')
    try:fs,samples=wavfile.read(io.BytesIO(raw))
    except (ValueError,OSError,EOFError):raise AcousticFailure('invalid_audio') from None
    if (samples.ndim not in (1,2) or not len(samples) or len(samples)>8_000_000 or
            samples.ndim==2 and not 1<=samples.shape[1]<=8):raise AcousticFailure('lpc_input_budget')
    if not 8000<=fs<=96000:raise AcousticFailure('lpc_sample_rate')
    first,last=int(settings.roi_start*fs),int(settings.roi_end*fs)
    if not 0<=first<last<=len(samples) or settings.roi_end>len(samples)/fs:raise AcousticFailure('lpc_invalid_roi')
    if last-first>MAX_ROI_SAMPLES:raise AcousticFailure('lpc_roi_budget')
    if not np.isfinite(samples).all():raise AcousticFailure('invalid_audio')
    audio=mono_samples(samples[first:last])
    label='';tier_name=None
    if textgrid is not None:
        from .io.annotations import decode_textgrid
        try:tiers=decode_textgrid(textgrid)
        except ValueError:raise AcousticFailure('invalid_textgrid') from None
        if not tiers:raise AcousticFailure('missing_or_invalid_tier')
        tier_name=settings.tier_name or tiers[0].name
        if not any(t.name==tier_name for t in tiers):raise AcousticFailure('missing_or_invalid_tier')
        if any(i.xmin<0 or i.xmax>len(samples)/fs+1/fs for t in tiers for i in t.intervals):
            raise AcousticFailure('lpc_textgrid_range')
        label=extract_label(tiers,tier_name,settings.roi_start,settings.roi_end)
        if len(label)>200:raise AcousticFailure('lpc_label_budget')
    elif settings.tier_name is not None:raise AcousticFailure('missing_or_invalid_tier')
    fonts=resolve_fonts(settings.font)
    numeric=LPCConfig(**{k:v for k,v in settings.model_dump().items() if k in LPCConfig.__dataclass_fields__})
    result=compute_spectrum(audio,int(fs),numeric)
    spectrum=LpcSpectrumData(frequencies_hz=result.frequencies_hz.tolist(),magnitude_db=result.magnitude_db.tolist(),
                            amp_min_db=result.amp_min_db,amp_max_db=result.amp_max_db)
    stream=io.BytesIO();wavfile.write(stream,int(fs),audio.astype(np.float64))
    metadata=dict(schema_version='m04/1',method_version='v2-lpc-autocorrelation/1',
        config=settings.model_dump(),input_name=input_name,input_sha256=digest(raw),
        textgrid_sha256=digest(textgrid) if textgrid is not None else None,
        sample_rate_hz=int(fs),sample_count=len(samples),channels=1 if samples.ndim==1 else samples.shape[1],
        channel_policy='arithmetic-mean; pcm-normalization; no-peak-normalization',
        selection=dict(start_sample=first,end_sample=last,start_s=first/fs,end_s=last/fs,interval='half-open'),
        label=label,tier_name=tier_name,spectrum=spectrum.model_dump(),runtime=runtime,render_fonts=fonts,
        source_ids=['REF-MAKHOUL-1975-LPC','M04-PY-NUMPY','M04-PY-SCIPY'],
        gain_policy='unit numerator; no prediction-error gain; not calibrated SPL',
        export_names=export_names(input_name,label,settings.roi_start,settings.roi_end))
    blobs={'lpc.ptb.json':json.dumps(metadata,ensure_ascii=False,allow_nan=False).encode(),
           'lpc_SPECTRUM.png':plot_bytes(result,settings,label,fonts),'lpc_AUDIO.wav':stream.getvalue()}
    header=dict(kind='prepared_lpc',audio_sha256=digest(raw),textgrid_sha256=metadata['textgrid_sha256'],
        files=[dict(name=n,format=n.rsplit('.',1)[1],size_bytes=len(blobs[n]),sha256=digest(blobs[n])) for n in LPC_NAMES])
    encoded=json.dumps(header).encode();payload=struct.pack('<Q',len(encoded))+encoded+b''.join(blobs[n] for n in LPC_NAMES)
    if len(payload)>OUTPUT_BYTES:raise AcousticFailure('analysis_output_limit')
    return payload


def run():
    from .acoustic_errors import AcousticFailure,public_error
    try:
        with open(sys.argv[1],'rb') as handle:
            header=json.loads(handle.readline(16384));raw=handle.read(INPUT_BYTES+GRID_BYTES+1)
        n=header['audio_size'];audio,grid=raw[:n],raw[n:]
        if (not 0<n<=INPUT_BYTES or len(grid)>GRID_BYTES or digest(audio)!=header['sha256'] or
                (digest(grid) if grid else None)!=header['textgrid_sha256']):raise AcousticFailure('invalid_audio')
        payload=prepare(audio,header['config'],header['input_name'],grid if header['textgrid_sha256'] else None)
    except Exception as exc:
        known={'invalid_audio':'invalid_audio','nonfinite_audio':'invalid_audio','invalid_sample_rate':'lpc_sample_rate',
            'invalid_roi':'lpc_invalid_roi','roi_too_large':'lpc_roi_budget','solver_failed':'lpc_solver_failed',
            'segment_too_short':'lpc_segment_too_short','egg_runtime_mismatch':'lpc_runtime_mismatch'}
        encoded=json.dumps({'error':known.get(getattr(exc,'code',None),public_error(exc))}).encode()
        payload=struct.pack('<Q',len(encoded))+encoded
    try:
        with open(sys.argv[2],'wb',buffering=0) as out:
            for offset in range(0,len(payload),65536):out.write(payload[offset:offset+65536])
    except OSError:raise SystemExit(2) from None
