"""Single-source memory session inside the pinned EGG compatibility runtime."""
import base64
import hashlib
import io
import json
import os
import sys


class Session:
    def __init__(self, raw=None, *, source_path=None, reaper=None, native_header=None):
        from scipy.io import wavfile
        from .egg_runtime import fingerprint
        from .spectrogram_preview import PreviewError
        fingerprint()
        self.recording = None
        if source_path:
            from .egg_recording import Recording
            source = Recording(source_path)
            if source.long:
                from phonetic_core.egg.bounded import TimeView
                self.recording = source
                self.fs, self.samples, self.sha = source.fs, TimeView(source.frames, source.fs), source.sha
            else:
                raw = source.path.read_bytes(); source.close()
        if self.recording is None:
            if not raw or not 0 < len(raw) <= 64_000_000: raise PreviewError('egg_input_budget')
            self.fs, self.samples = wavfile.read(io.BytesIO(raw))
            self.sha = hashlib.sha256(raw).hexdigest()
        if self.recording is None:
            self.validate_short()
        self.prepared_key = self.events_key = self.audio_key = None
        self.result = None
        self.main = {}
        from collections import OrderedDict
        self.pitch_cache=OrderedDict()
        self.reaper = reaper
        self.native_header = native_header or {}

    def validate_short(self):
        from .spectrogram_preview import PreviewError
        if self.samples.ndim != 2 or self.samples.shape[1] != 2: raise PreviewError('egg_stereo_required')
        if not 8000 <= self.fs <= 96000: raise PreviewError('egg_sample_rate')
        if not 0 < len(self.samples) <= 5_760_000 or len(self.samples)/self.fs > 120: raise PreviewError('egg_input_budget')

    def update(self, config):
        from phonetic_core.egg import EGGConfig, prepare, analyze_events
        from .egg_f0 import populate, native_engine
        from ptb_api.egg_models import EggTaskConfig
        from .spectrogram_preview import PreviewError
        from .egg_preview import preview_files
        settings = EggTaskConfig.model_validate(config)
        if settings.mode != 'preview' or settings.glottal_movement: raise PreviewError('egg_invalid_preview')
        first = int(settings.roi_start*self.fs)
        last = len(self.samples) if settings.roi_end is None else int(settings.roi_end*self.fs)
        if not 0 <= first < last <= len(self.samples): raise PreviewError('egg_invalid_roi')
        if self.recording and last-first > self.fs*60: raise PreviewError('egg_viewport_budget')
        numerical = EGGConfig.for_workbench(**{k:v for k,v in settings.model_dump().items() if k in EGGConfig.__dataclass_fields__})
        key = (settings.flip_channels, settings.highpass_cutoff, settings.lowpass_cutoff, settings.f0_policy)
        if key != self.prepared_key:
            result = (self.recording.result(numerical, settings.flip_channels) if self.recording else
                      prepare(self.samples, int(self.fs), numerical, flip_channels=settings.flip_channels))
            if self.recording and self.result is not None and self.prepared_key[0] == key[0] and self.prepared_key[3] == key[3]:
                for name in ('audio_f0_times','audio_f0_values','reaper_f0_times','reaper_f0_values'):
                    setattr(result,name,getattr(self.result,name))
            self.result, self.prepared_key, self.events_key = result, key, None
            self.main.clear()
        result = self.result
        events_key = tuple(getattr(numerical,name) for name in (
            'highpass_cutoff','lowpass_cutoff','gci_method','goi_method','criterion_level',
            'peak_prominence','valley_prominence','auto_prominence','min_auto_prominence'))
        if self.recording:events_key+=(first,last)
        if settings.keep_gci_f0 and events_key != self.events_key:
            if self.recording:
                from phonetic_core.egg.bounded import analyze_events as bounded_events
                result=self.result=bounded_events(result,numerical,span=(first,last))
            else: result = self.result = analyze_events(result, numerical)
            self.events_key = events_key
            self.main.clear()
        pitch_key=(settings.flip_channels,settings.f0_policy,first,last)
        if self.recording:
            cached=self.pitch_cache.get(pitch_key,{})
            for prefix in ('audio','reaper'):
                for suffix in ('times','values'):
                    setattr(result,prefix+'_f0_'+suffix,cached.get(prefix+'_f0_'+suffix))
        if ((settings.keep_praat_f0 and result.audio_f0_times is None) or
                (settings.keep_reaper_f0 and result.reaper_f0_times is None)):
            header=self.native_header if settings.keep_reaper_f0 and result.reaper_f0_times is None else {}
            with native_engine(header) as native:
                populate(result,settings,self.reaper or native,praat=settings.keep_praat_f0,
                         span=(first,last) if self.recording else None)
            self.main.clear()
        if self.recording:
            self.pitch_cache[pitch_key]={name:getattr(result,name) for name in (
                'audio_f0_times','audio_f0_values','reaper_f0_times','reaper_f0_values')}
            self.pitch_cache.move_to_end(pitch_key)
            while len(self.pitch_cache)>3:self.pitch_cache.popitem(last=False)
        # Disabled tracks must match the old non-requested preview exactly.
        praat = result.audio_f0_times, result.audio_f0_values
        if not settings.keep_praat_f0: result.audio_f0_times = result.audio_f0_values = None
        audio_key = (settings.flip_channels,first,last) if self.recording else settings.flip_channels
        try:
            data, blobs = preview_files(result, numerical, settings, first, last,
                cache=self.main, include_audio=self.audio_key != audio_key)
        finally:
            result.audio_f0_times, result.audio_f0_values = praat
        self.audio_key = audio_key
        return dict(input_sha256=self.sha, sample_rate_hz=int(self.fs), sample_count=len(self.samples),
            config=settings.model_dump(), selection=dict(start_s=first/self.fs, end_s=last/self.fs), preview=data,
            psd_base64=base64.b64encode(blobs['egg_PSD.png']).decode(),
            audio_start_s=first/self.fs if self.recording else 0.,
            audio_base64=base64.b64encode(blobs['egg_AUDIO.wav']).decode() if 'egg_AUDIO.wav' in blobs else None)


def run():
    # Wait for the owner to bind this PID to its resource Job before imports.
    print(os.getpid(), flush=True)
    session = None
    while line := sys.stdin.buffer.readline(32768):
        try:
            header = json.loads(line)
            if 'source_path' in header:
                session = Session(source_path=header['source_path'],native_header=header)
                value = dict(sha256=session.sha,sample_rate_hz=int(session.fs),sample_count=len(session.samples),
                    overview_base64=base64.b64encode(session.recording.overview()).decode() if session.recording else None)
            elif 'size' in header:
                size = header['size']
                if type(size) is not int or not 0 < size <= 64_000_000: raise ValueError()
                raw = sys.stdin.buffer.read(size)
                if len(raw) != size: raise ValueError()
                session = Session(raw,native_header=header)
                value = dict(sha256=session.sha)
            elif session is not None:
                value = session.update(header['config'])
            else: raise ValueError()
        except Exception as exc:
            code = getattr(exc, 'code', '')
            known = {'invalid_audio':'invalid_audio','invalid_signal':'invalid_audio','filter_cutoff':'egg_filter_failed','invalid_filter_cutoff':'egg_filter_failed',
                     'filter_failed':'egg_filter_failed','invalid_roi':'egg_invalid_roi'}
            value = dict(error=known.get(code, code if code.startswith('egg_') else 'egg_preview_failed'))
        payload = json.dumps(value, allow_nan=False).encode()
        if len(payload) > 64_000_000: raise SystemExit(2)
        sys.stdout.buffer.write(str(len(payload)).encode()+b'\n'+payload)
        sys.stdout.buffer.flush()
