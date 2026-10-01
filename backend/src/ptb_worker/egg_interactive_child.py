"""Single-source memory session inside the pinned EGG compatibility runtime."""
import base64
import hashlib
import io
import json
import os
import sys


class Session:
    def __init__(self, raw):
        from scipy.io import wavfile
        from .egg_runtime import fingerprint
        from .spectrogram_preview import PreviewError
        fingerprint()
        if not 0 < len(raw) <= 64_000_000: raise PreviewError('egg_input_budget')
        self.fs, self.samples = wavfile.read(io.BytesIO(raw))
        if self.samples.ndim != 2 or self.samples.shape[1] != 2: raise PreviewError('egg_stereo_required')
        if not 8000 <= self.fs <= 96000: raise PreviewError('egg_sample_rate')
        if not 0 < len(self.samples) <= 5_760_000 or len(self.samples)/self.fs > 120: raise PreviewError('egg_input_budget')
        self.sha = hashlib.sha256(raw).hexdigest()
        self.prepared_key = self.events_key = self.audio_key = None
        self.result = None
        self.main = {}

    def update(self, config):
        from phonetic_core.egg import EGGConfig, prepare, analyze_events
        from phonetic_core.egg.f0 import praat_pitch
        from ptb_api.egg_models import EggTaskConfig
        from .spectrogram_preview import PreviewError
        from .egg_preview import preview_files
        settings = EggTaskConfig.model_validate(config)
        if settings.mode != 'preview' or settings.glottal_movement: raise PreviewError('egg_invalid_preview')
        first = int(settings.roi_start*self.fs)
        last = len(self.samples) if settings.roi_end is None else int(settings.roi_end*self.fs)
        if not 0 <= first < last <= len(self.samples): raise PreviewError('egg_invalid_roi')
        numerical = EGGConfig.for_workbench(**{k:v for k,v in settings.model_dump().items() if k in EGGConfig.__dataclass_fields__})
        key = (settings.flip_channels, settings.highpass_cutoff, settings.lowpass_cutoff)
        if key != self.prepared_key:
            result = prepare(self.samples, int(self.fs), numerical, flip_channels=settings.flip_channels)
            self.result, self.prepared_key, self.events_key = result, key, None
            self.main.clear()
        result = self.result
        events_key = repr(numerical)
        if settings.keep_gci_f0 and events_key != self.events_key:
            result = self.result = analyze_events(result, numerical)
            self.events_key = events_key
            self.main.clear()
        if settings.keep_praat_f0 and result.audio_f0_times is None:
            track = praat_pitch(result.audio_signal, int(self.fs))
            result.audio_f0_times, result.audio_f0_values = track.times, track.values
            self.main.clear()
        # Disabled tracks must match the old non-requested preview exactly.
        praat = result.audio_f0_times, result.audio_f0_values
        if not settings.keep_praat_f0: result.audio_f0_times = result.audio_f0_values = None
        try:
            data, blobs = preview_files(result, numerical, settings, first, last,
                cache=self.main, include_audio=self.audio_key != settings.flip_channels)
        finally:
            result.audio_f0_times, result.audio_f0_values = praat
        self.audio_key = settings.flip_channels
        return dict(input_sha256=self.sha, sample_rate_hz=int(self.fs), sample_count=len(self.samples),
            config=settings.model_dump(), selection=dict(start_s=first/self.fs, end_s=last/self.fs), preview=data,
            psd_base64=base64.b64encode(blobs['egg_PSD.png']).decode(),
            audio_base64=base64.b64encode(blobs['egg_AUDIO.wav']).decode() if 'egg_AUDIO.wav' in blobs else None)


def run():
    # Wait for the owner to bind this PID to its resource Job before imports.
    print(os.getpid(), flush=True)
    session = None
    while line := sys.stdin.buffer.readline(32768):
        try:
            header = json.loads(line)
            if 'size' in header:
                size = header['size']
                if type(size) is not int or not 0 < size <= 64_000_000: raise ValueError()
                raw = sys.stdin.buffer.read(size)
                if len(raw) != size: raise ValueError()
                session = Session(raw)
                value = dict(sha256=session.sha)
            elif session is not None:
                value = session.update(header['config'])
            else: raise ValueError()
        except Exception as exc:
            code = getattr(exc, 'code', '')
            known = {'filter_cutoff':'egg_filter_failed','invalid_filter_cutoff':'egg_filter_failed',
                     'filter_failed':'egg_filter_failed','invalid_roi':'egg_invalid_roi'}
            value = dict(error=known.get(code, code if code.startswith('egg_') else 'egg_preview_failed'))
        payload = json.dumps(value, allow_nan=False).encode()
        if len(payload) > 64_000_000: raise SystemExit(2)
        sys.stdout.buffer.write(str(len(payload)).encode()+b'\n'+payload)
        sys.stdout.buffer.flush()
