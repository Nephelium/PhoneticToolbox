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


def populate(result,settings,reaper=None,*,praat=False,span=None):
    from phonetic_core.egg.f0 import praat_pitch,reaper_pitch
    if result.method_version == 'egg-bounded/2':
        import numpy as np
        low,high=bounds(settings)
        for requested,prefix,compute in ((praat,'audio',lambda y:praat_pitch(y,result.fs,pitch_floor=low,pitch_ceiling=high)),
            (settings.keep_reaper_f0,'reaper',lambda y:reaper_pitch(y,result.fs,reaper))):
            if not requested or getattr(result,prefix+'_f0_times') is not None:continue
            if prefix=='reaper' and reaper is None:raise AcousticFailure('egg_reaper_unavailable')
            times=[];values=[];fs=result.fs;frames=len(result.time_vector)
            begin,end=span if span is not None else (0,frames)
            for first in range(begin,end,fs*20):
                last=min(end,first+fs*20);left=max(0,first-fs);right=min(frames,last+fs)
                track=compute(result.audio_signal[left:right])
                t=track.times+left/fs;take=(t>=first/fs)&(t<last/fs)
                times.extend(t[take]);values.extend(track.values[take])
            setattr(result,prefix+'_f0_times',np.asarray(times));setattr(result,prefix+'_f0_values',np.asarray(values))
        return
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
