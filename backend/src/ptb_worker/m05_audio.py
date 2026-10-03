"""Timestamp-audited audio export. Original encoded input remains authoritative."""
from pathlib import Path
import wave
import json
import math

def extract_audio(source,directory,*,stop=lambda:False,max_bytes=128_000_000,clock_policy='container_pts'):
    import av
    import numpy as np
    if clock_policy not in ('container_pts','decoded_samples'):raise ValueError('m05_audio_clock_policy')
    directory=Path(directory)
    with av.open(str(source)) as container:
        if not container.streams.audio:return dict(present=False),[]
        stream=container.streams.audio[0];rate=stream.codec_context.sample_rate
        channels=len(stream.codec_context.layout.channels)
        if not 8000<=rate<=192000 or channels not in (1,2):raise ValueError('m05_audio_format_limit')
        resampler=av.AudioResampler(format='s16',layout='mono' if channels==1 else 'stereo',rate=rate)
        anchor=None;written=0;last=None;gaps=0;count=0;max_residual=0;last_residual=0;records=directory/'audio-timing.jsonl'
        with wave.open(str(directory/'audio_recording.wav'),'wb') as output,records.open('x',encoding='utf8') as timeline:
            output.setparams((channels,2,rate,0,'NONE','not compressed'))
            for frame in container.decode(stream):
                if stop():raise InterruptedError('cancelled')
                if frame.pts is None or frame.time_base is None:raise ValueError('m05_audio_missing_pts')
                t=float(frame.pts*frame.time_base)
                if last is not None and t<=last:raise ValueError('m05_audio_nonmonotonic_pts')
                last=t;anchor=t if anchor is None else anchor
                for converted in resampler.resample(frame):
                    if converted.pts is None:raise ValueError('m05_audio_missing_resampled_pts')
                    start=round((float(converted.pts*converted.time_base)-anchor)*rate)
                    gap=start-written
                    residual=gap
                    last_residual=residual
                    max_residual=max(max_residual,abs(residual))
                    # A container tick rounds a packet timestamp, whereas its
                    # decoded sample count stays exact. Keep both observations.
                    # A difference of two rounded PTS (current minus anchor)
                    # has up to one whole container tick of uncertainty.
                    quantization=math.ceil(float(frame.time_base)*rate)+1
                    if abs(gap)<=quantization:gap=0
                    # Browser recording export preserves every decoded PCM sample.
                    # MediaRecorder PTS can jitter relative to the sample clock;
                    # retain that discrepancy in the sidecar, without adding or
                    # dropping audio samples. Formal arbitrary-input analysis keeps
                    # the strict container-PTS policy by default.
                    if clock_policy=='decoded_samples':gap=0
                    if gap < -1:raise ValueError('m05_audio_overlapping_samples')
                    payload=converted.to_ndarray().astype('<i2',copy=False).tobytes()
                    if (written+max(0,gap)+converted.samples)*channels*2+44>max_bytes:raise ValueError('m05_audio_output_budget')
                    if gap>0:
                        remaining=gap*channels*2
                        while remaining:
                            n=min(65536,remaining);output.writeframesraw(bytes(n));remaining-=n
                        written+=gap;gaps+=gap
                    output.writeframesraw(payload)
                    timeline.write(json.dumps(dict(index=count,pts=frame.pts,time_base=[frame.time_base.numerator,frame.time_base.denominator],media_time_s=t,output_sample_start=written,samples=converted.samples,inserted_silence_samples=max(0,gap),timestamp_residual_samples=residual,timestamp_rounding_bound_samples=quantization))+'\n')
                    written+=converted.samples;count+=1
            # Same sample rate/format resampling must not leave buffered samples.
            if resampler.resample(None):raise ValueError('m05_audio_unflushed_samples')
    return dict(present=True,sample_rate=rate,channels=channels,decoded_frames=count,samples=written,
                first_decoded_pts_s=anchor,end_s=anchor+written/rate if anchor is not None else None,
                inserted_silence_samples=gaps,encoding='PCM16 decoded derivative',
                drift_s=None,clock=clock_policy,max_timestamp_residual_samples=max_residual,last_timestamp_residual_samples=last_residual,
                timestamp_vs_pcm_end_residual_s=last_residual/rate,physical_sync_verified=False),['audio_recording.wav','audio-timing.jsonl']
