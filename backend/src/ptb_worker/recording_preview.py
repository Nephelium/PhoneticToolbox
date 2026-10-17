"""Bounded, read-only M16 Praat display using the desktop display core."""
import base64
import json
import os
from pathlib import Path
import sys
from .spectrogram_preview import PreviewError
from .preview_cache import memoized_preview


def display(payload,*,start,end,channel,width):
    from .recording_child import load
    from phonetic_core.recording.praat_display import PraatDisplaySpectrum
    x,rate=load(payload)
    if type(start)!=int or type(end)!=int or not 0<=start<end<=len(x):raise ValueError('录音预览选区无效')
    engine=PraatDisplaySpectrum(end-start,rate,channel,width);engine.add(x[start:end]);result=engine.result()
    pixels=result.pop('pixels');result.pop('rows')
    result.update(columns=len(pixels),pixels_base64=base64.b64encode(bytes(p for row in pixels for p in row)).decode('ascii'))
    return result


@memoized_preview('recording')
def render(payload,*,start,end,channel,width=640):
    if len(payload)>100_000_000:raise PreviewError('preview_too_large',413)
    if sys.platform!='linux':raise PreviewError('recording_server_unavailable',503)
    from .native.linux_runtime import command,load_profile
    from .native.posix import run_bounded
    from .io.limits import Limits,LimitError,FormatError
    header=json.dumps(dict(size=len(payload),start=start,end=end,channel=channel,width=width)).encode()+b'\n'
    try:
        _,profile=load_profile()
        raw=run_bounded(command('recording_preview','--preview'),header+payload,Path(profile['cache']),Limits(input_bytes=100_010_000,output_bytes=512_000,process_bytes=800_000_000,timeout_seconds=30))
        result=json.loads(raw)
        if 'error' in result:raise PreviewError('invalid_spectrogram_input')
        return result
    except (LimitError,FormatError,OSError,ValueError) as exc:
        if isinstance(exc,PreviewError):raise
        raise PreviewError('preview_resource_failed',503) from None


def child():
    try:
        header=json.loads(sys.stdin.buffer.readline(8192));size=header.pop('size')
        if type(size)!=int or not 0<size<=100_000_000:raise ValueError('size')
        payload=sys.stdin.buffer.read(size)
        if len(payload)!=size:raise ValueError('truncated input')
        result=display(payload,**header)
    except Exception:result={'error':'invalid_spectrogram_input'}
    sys.stdout.buffer.write(json.dumps(result,allow_nan=False).encode());sys.stdout.buffer.flush()


if __name__=='__main__':child()
