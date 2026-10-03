"""M03-R5 shared F0 orchestration for interactive and persistent tasks."""
from contextlib import contextmanager
from .acoustic_errors import AcousticFailure

NATIVE_BYTES = 4_000_000  # 120 s at 16 kHz PCM16 plus WAV header.


def bounds(settings):
    return (30.,800.) if settings.f0_policy == 'audio-f0/2' else (75.,600.)


@contextmanager
def native_engine(header):
    if not header.get('native_scratch') or not header.get('reaper_binary'):
        yield None
        return
    from .managed_scratch import ReservedNativeScratch
    from .native.reaper import Reaper
    from .io.limits import Limits
    with ReservedNativeScratch(header['native_scratch'],NATIVE_BYTES) as scratch:
        try:
            native=Reaper(header['reaper_binary'],scratch,Limits(input_bytes=NATIVE_BYTES,
                samples=5_760_000,output_bytes=2_000_000,process_bytes=1_000_000_000,timeout_seconds=25))
        except (ValueError,OSError):raise AcousticFailure('egg_reaper_unavailable') from None
        yield native


def populate(result,settings,reaper=None,*,praat=False):
    from phonetic_core.egg.f0 import praat_pitch,reaper_pitch
    if praat and result.audio_f0_times is None:
        low,high=bounds(settings)
        track=praat_pitch(result.audio_signal,result.fs,pitch_floor=low,pitch_ceiling=high)
        result.audio_f0_times,result.audio_f0_values=track.times,track.values
    if settings.keep_reaper_f0 and result.reaper_f0_times is None:
        if reaper is None:raise AcousticFailure('egg_reaper_unavailable')
        try:track=reaper_pitch(result.audio_signal,result.fs,reaper)
        except Exception as exc:
            from .io.limits import Cancelled,LimitError
            if isinstance(exc,(Cancelled,LimitError)):raise
            raise AcousticFailure('egg_reaper_failed') from exc
        result.reaper_f0_times,result.reaper_f0_values=track.times,track.values


def evidence(settings,reaper=None):
    low,high=bounds(settings)
    return dict(revision=settings.f0_policy,praat=dict(method='autocorrelation',floor_hz=low,ceiling_hz=high,time_step_s=.01),
        reaper=(dict(backend='native_reaper',binary_sha256=reaper.sha256,floor_hz=30.,ceiling_hz=800.,
            time_step_s=.01,resample_hz=16000,hilbert=False,highpass=True,time_axis='native EST')
            if settings.keep_reaper_f0 and reaper is not None else None))
