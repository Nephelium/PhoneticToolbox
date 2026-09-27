"""M08 scientific child handler. Caller supplies an authorized input and bounded writer.

No standalone host, executor, credentials, DB migration or direct user-file writes.
The platform must publish returned artifacts through its fenced quota transaction.
"""
import hashlib
from pathlib import Path
import numpy as np
import parselmouth
from ptb_api.m08_models import M08Config
from phonetic_core.manipulation.m08_synthesis import synthesize_from_pitch
from phonetic_core.manipulation.m08_transform import transform
from phonetic_core.manipulation.m08_batch import generate_batch_linear
from phonetic_core.manipulation.m08_rules import track, validate_range, validate_controls, next_name


def execute(sound, config, stem, *, cancelled=lambda: False, emit=None):
    try:
        return _execute(sound,config,stem,cancelled=cancelled,emit=emit)
    except parselmouth.PraatError as error:
        raise ValueError('m08_praat_error') from error


def _execute(sound, config, stem, *, cancelled=lambda: False, emit=None):
    """Emit serially, never aggregate batch audio in RAM. Success requires caller commit.

    emit receives a suggested name (not a path), Sound and immutable snapshot data.
    Caller reserves/allocates collision-free names atomically. Cancellation is checked
    between native calls; hard cancellation/time/memory limits belong to executor.
    """
    config = M08Config.model_validate(config)
    if not stem or any(c in stem for c in '/\\\0') or len(stem) > 100:
        raise ValueError('m08_invalid_stem')
    if sound.n_samples > 8_000_000 or sound.n_channels > 8:
        raise ValueError('m08_input_budget')
    if not np.isfinite(sound.values).all(): raise ValueError('m08_nonfinite_audio')
    def check():
        if cancelled(): raise InterruptedError('m08_cancelled')
    check()
    start, end = config.start, config.end if config.end is not None else sound.xmax
    validate_range(sound, start, end)
    if (end-start)*sound.sampling_frequency/min(config.speed,1) > 32_000_000:
        raise ValueError('m08_output_budget')
    times, f0 = track(sound)
    info = dict(schema_version='m08/1', backend='Praat/Parselmouth', backend_version=parselmouth.__version__,
                source_ids=['SRC-PRAAT'], config=config.model_dump(), source_start_s=start, source_end_s=end,
                sample_rate_hz=sound.sampling_frequency, source_frames=sound.n_samples,
                times=times.tolist(), original_f0=f0.tolist())
    if config.action == 'preview': return info
    if emit is None: raise ValueError('m08_writer_required')
    check()
    if config.action == 'synthesize':
        modified = np.asarray(config.modified_f0, dtype=float)
        if modified.shape != f0.shape: raise ValueError('m08_curve_length')
        outputs = [(next_name(stem,start,end,[]), synthesize_from_pitch(sound,times,modified,start,end,config.speed), ())]
    elif config.action == 'transform':
        # V2 folder transform always processes the entire file.
        if start != sound.xmin or end != sound.xmax: raise ValueError('m08_transform_requires_whole')
        # M08-FIX01: Praat 0.4.7 accepts "Hertz", rejects V2's "Hz".
        outputs = [(stem+'.wav', transform(sound,config.speed,config.pitch_ratio,config.pitch_hz,hz_unit='Hertz'), ())]
    else:
        points = [p.model_dump() for p in config.points]
        validate_controls(points,start,end)
        outputs = generate_batch_linear(sound,stem,times,f0,start,end,points[0]['time'],points[-1]['time'],
            points[0]['freqs'],points[-1]['freqs'],points[1:-1],points[0]['mode'],points[-1]['mode'],
            [p['mode'] for p in points[1:-1]],config.offset)
    results = []
    for name, output, controls in outputs:
        check()
        # History follows V2: re-extract pitch from saved PCM16 audio, with relative time.
        if output.n_samples > 32_000_000: raise ValueError('m08_output_budget')
        metadata = dict(info, output_frames=output.n_samples, output_start_s=output.xmin,
                        output_end_s=output.xmax, controls=list(controls))
        reference = emit(name,output,metadata)
        results.append(reference)
        check()
    return dict(info, outputs=results)


def read_authorized(path, sha256):
    """Only call after owner/expiry and path-resolution checks by shared storage."""
    path = Path(path)
    if path.stat().st_size > 64_000_000: raise ValueError('m08_input_budget')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != sha256: raise ValueError('m08_input_changed')
    try:
        return parselmouth.Sound(str(path))
    except parselmouth.PraatError as error:
        raise ValueError('m08_audio_decode_failed') from error
