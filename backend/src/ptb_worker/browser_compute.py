"""Fixed browser Worker entry points, reusing the server/core implementations.

The browser has no native filesystem grants or server runtime profile. Its
host pins the downloaded source/archive/runtime hashes instead. Server callers
retain their original native runtime validation and process/resource boundary.
"""
import base64
import hashlib
import json
import os

os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS','25000000')
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_WIDTH','1000000')
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_HEIGHT','1000000')

sessions = {}


def browser_runtime():
    import sys
    import numpy
    import scipy
    if sys.platform!='emscripten' or numpy.__version__!='2.2.5' or scipy.__version__!='1.14.1':
        raise RuntimeError('Browser runtime identity mismatch')


def execute(operation, raw, args):
    if operation=='lpc_fonts':
        from .fonts import check_fonts
        from ptb_api.font_models import FigureFontSnapshot
        return check_fonts(FigureFontSnapshot.model_validate(args))
    if operation=='lpc':
        import struct
        from .lpc_child import prepare
        from ptb_api.protocol_version import CORE_VERSION
        grid=base64.b64decode(args['textgrid'],validate=True) if args.get('textgrid') is not None else None
        if grid is not None and len(grid)>2_000_000:raise ValueError('invalid_textgrid')
        def identity():
            browser_runtime()
            return dict(versions={'phonetic-core':CORE_VERSION,'numpy':'2.2.5','scipy':'1.14.1'},platform='browser',runtime='Pyodide 0.29.3')
        payload=prepare(raw,args['config'],args['name'],grid,runtime_check=identity)
        length=struct.unpack('<Q',payload[:8])[0];header=json.loads(payload[8:8+length]);offset=8+length;files=[]
        for item in header['files']:
            value=payload[offset:offset+item['size_bytes']];offset+=len(value)
            if hashlib.sha256(value).hexdigest()!=item['sha256']:raise ValueError('LPC output hash mismatch')
            files.append(dict(name=item['name'],data=base64.b64encode(value).decode()))
        if offset!=len(payload):raise ValueError('LPC output framing mismatch')
        return dict(files=files)
    if operation=='phonology':
        from .m14_jobs import execute as phonology
        from ptb_api.m14_models import M14Config
        config=M14Config.model_validate(args['config']).model_dump()
        outputs=phonology(raw,args['name'],config)
        if sum(map(len,outputs.values()))>16_000_000:raise ValueError('m14_output_budget')
        return dict(files=[dict(name=name,data=base64.b64encode(value).decode()) for name,value in outputs.items()])
    if operation=='speech_synthesis':
        from .m06_child import compute
        if args['action'] not in ('generate','synthesize'):raise ValueError('Browser extraction is unavailable')
        if len(args['parameters'].encode())>2_000_000:raise ValueError('m06_input_budget')
        from phonetic_core.synthesis.klatt.api import import_parameters
        config=import_parameters(args['parameters'])
        if args['action']=='synthesize' and config['duration']>3:raise ValueError('Browser synthesis duration budget')
        if type(args['seed']) is not int or not 0<=args['seed']<=0xffffffff:raise ValueError('Invalid synthesis seed')
        header=dict(action=args['action'],parameters=args['parameters'],seed=args['seed'],input_sha256=hashlib.sha256(raw).hexdigest())
        outputs=compute(header,raw)
        return dict(files=[dict(name=name,data=base64.b64encode(value).decode()) for name,value in outputs])
    if operation=='parameters':
        from .io.legacy_parameters import read_legacy_parameters
        if len(raw)>16_000_000:raise ValueError('parameter_input_budget')
        return dict(sha256=hashlib.sha256(raw).hexdigest(),**read_legacy_parameters(raw,args['name']))
    if operation=='prediction':
        import math
        from .spec2wav_child import preview
        result=preview(raw,args)
        if 'prediction' in result:
            window=result['prediction']['predicted_window_ms']
            # Float32 CNN kernels vary slightly across BLAS builds. Use the
            # canonical server when this can change the rounded integer window.
            if abs(window-(math.floor(window)+.5))<=.001:
                raise ValueError('Browser prediction requires canonical rounding')
        return result
    if operation=='reconstruction':
        from .spec2wav_child import prepare
        from .segmentation import unpack_bundle
        if args.get('algorithm','griffin_lim')!='griffin_lim' or args.get('vocoder','none')!='none':
            raise ValueError('Browser reconstruction requires Griffin-Lim')
        bundle=unpack_bundle(prepare(raw,args),64_000_000)
        return dict(files=[dict(name=spec['name'],data=base64.b64encode(payload).decode()) for spec,payload in zip(bundle.manifest['files'],bundle.payloads)])
    if operation=='spectrogram':
        from .io.audio import decode_wav
        from .io.limits import Limits
        from phonetic_core.spectrogram import spectrogram_preview
        audio=decode_wav(raw,Limits(input_bytes=64_000_000,samples=32_000_000,channels=32))
        result=spectrogram_preview(audio,**args)
        result.pop('_power');result.pop('_frequencies')
        result['pixels_base64']=base64.b64encode(result.pop('pixels')).decode()
        result['sha256']=hashlib.sha256(raw).hexdigest()
        return result
    if operation=='recording_spectrum':
        from .recording_child import load
        from phonetic_core.recording.praat_display import PraatDisplaySpectrum
        x,rate=load(raw);start,end=args['start'],args['end']
        if type(start)!=int or type(end)!=int or not 0<=start<end<=len(x):raise ValueError('Invalid recording viewport')
        engine=PraatDisplaySpectrum(end-start,rate,args['channel'],args.get('width',640))
        engine.add(x[start:end]);result=engine.result();pixels=result.pop('pixels');result.pop('rows')
        result.update(columns=len(pixels),pixels_base64=base64.b64encode(bytes(p for row in pixels for p in row)).decode())
        return result
    if operation=='egg_open':
        from .egg_interactive_child import Session
        # One decoded EGG source per tab, bounded by the original 64 MB/120 s.
        sessions.clear();session=Session(raw,runtime_check=browser_runtime)
        sessions[args['id']]=session
        return dict(session_id=args['id'],sha256=session.sha,sample_rate_hz=int(session.fs),sample_count=len(session.samples))
    if operation=='egg_close':
        sessions.pop(args['id'],None);return dict(closed=True)
    if operation=='egg_update':
        session=sessions.get(args['id'])
        if session is None:raise ValueError('egg_preview_expired')
        config=args['config']
        if config.get('keep_reaper_f0'):raise ValueError('Browser REAPER is unavailable')
        if args.get('needAudio'):session.audio_key=None
        result=session.update(config)
        suggested=result['preview'].get('suggested_db_range')
        if args.get('autoDb') and suggested:
            audio=result['audio_base64']
            result=session.update({**config,'spec_vmin':suggested[0],'spec_vmax':suggested[1]})
            result['audio_base64']=audio
        return result
    if operation=='recording_compute':
        from .recording_child import compute
        from ptb_api.recording_models import RecordingConfig
        payloads=[raw]+[base64.b64decode(v,validate=True) for v in args.get('extra',[])]
        if sum(map(len,payloads))>200_000_000:raise ValueError('Recording input budget')
        config=RecordingConfig.model_validate(args['config']).model_dump()
        wav,metadata=compute(payloads,config)
        return dict(audio_base64=base64.b64encode(wav).decode() if wav else None,metadata=metadata)
    raise ValueError('Unsupported browser operation')


def dispatch(operation, raw, arguments):
    result=execute(operation,bytes(raw.to_py()) if hasattr(raw,'to_py') else bytes(raw),json.loads(arguments))
    return json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))
