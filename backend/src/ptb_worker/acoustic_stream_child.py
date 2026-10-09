"""Owned M01 disk-streaming child. Scientific arrays remain block bounded."""
import os
import sys
import json
import time
from pathlib import Path

def atomic(path,value):
    temp=path.with_suffix('.next')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    # A Windows reader may briefly hold the old progress file without delete
    # sharing. Keep atomic publication and retry that short sharing conflict.
    for attempt in range(51):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            if attempt==50:raise
            time.sleep(.02)

def run(root,request):
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    import tempfile
    tempfile.tempdir=str(root)
    if request.get('operation')=='textgrid_segment':
        from .acoustic_stream_segments import run_segments
        return run_segments(root,request)
    import soundfile as sf
    from phonetic_core.services.acoustic_stream import iter_parameters,validate_source
    from phonetic_core.models.associations import AcousticAssociations
    from phonetic_core.ports.acoustic import AcousticBackends
    from .io.parameter_bundle import BundleWriter,export_xlsx,digest
    from .io.annotations import decode_textgrid
    from .io.lip import decode_lip
    from .io.limits import Limits
    from .io.scratch import Scratch
    from .native.reaper import Reaper
    from .reaper_policy import PolicyReaper
    from .acoustic_result import to_core_config,SOURCE_IDS
    from ptb_api.acoustic_models import AcousticConfigSnapshot,config_digest
    from phonetic_core import __version__ as core_version
    config=AcousticConfigSnapshot.model_validate(request['config'])
    options=config.extended.model_dump();index=str(request['index'])
    if index in options['channel_overrides']:
        options['audio_channel']=options['channel_overrides'][index]
        if options['egg']:options['egg']['egg_channel']=1-options['audio_channel']
    def progress(stage,fraction):
        bounds={'scan':(.02,.08),'egg':(.1,.15),'analysis':(.25,.5),'xlsx':(.75,.18)}
        first,span=bounds[stage]
        atomic(root/'status.json',dict(stage=stage,fraction=fraction,progress=first+span*fraction))
    def check():
        if (root/'cancel').exists():raise ValueError('cancelled')
    source=root/'input.wav'
    with sf.SoundFile(source) as audio:
        validate_source(audio.frames,audio.samplerate,audio.channels)
        def read(first,last):
            audio.seek(first)
            return audio.read(last-first,dtype='float64',always_2d=True)
        associations=AcousticAssociations(
            tiers=decode_textgrid((root/'textgrid.bin').read_bytes()) if (root/'textgrid.bin').exists() else (),
            lip=decode_lip((root/'lip.bin').read_bytes()) if (root/'lip.bin').exists() else None)
        scratch=Scratch(root,128_000_000)
        limits=Limits(input_bytes=128_000_000,samples=32_000_000,output_bytes=128_000_000,process_bytes=1_500_000_000,timeout_seconds=180)
        backend=PolicyReaper(config.backend_policy.reaper,lambda:Reaper(request['reaper_binary'],scratch,limits)) if config.backend_policy.reaper!='disabled' else None
        writer=BundleWriter(root/'result.ptb.sqlite');events=[];metadata={}
        try:
            for kind,arrays,info in iter_parameters(read,audio.frames,audio.samplerate,audio.channels,
                    to_core_config(config),options,AcousticBackends(reaper=backend),associations,progress,check):
                writer.append(kind,arrays);metadata=info
                events.extend(info.get('backends',[]))
            metadata.update(config=config.model_dump(),audio_sha256=request['audio_sha256'],
                core_version=core_version,config_sha256=config_digest(config),
                source_ids=[*SOURCE_IDS,*(['PENDING-EGG'] if options['egg'] else [])],
                backend_observations=[json.loads(v) for v in sorted({json.dumps(e,sort_keys=True) for e in events})],
                reaper_sha256=backend.native_sha256 if backend else None)
            if not writer.tables.get('params',{}).get('rows'):raise ValueError('no_parameter_frames')
            metadata=writer.finish(metadata)
        finally:
            writer.conn.close();scratch.close()
    export_xlsx(root/'result.ptb.sqlite',root/'result.xlsx',progress,check)
    metadata['artifacts']={n:dict(size_bytes=(root/n).stat().st_size,sha256=digest(root/n)) for n in ('result.xlsx','result.ptb.sqlite')}
    atomic(root/'result.ptb.json',metadata)
    return dict(success=True,files=['result.xlsx','result.ptb.sqlite','result.ptb.json'])

if __name__=='__main__':
    root=Path(sys.argv[1]).absolute()
    try:response=run(root,json.loads((root/'request.json').read_text('utf-8')))
    except Exception as error:
        import traceback
        from .acoustic_errors import public_error
        code=str(error) if str(error).startswith('m01_') else public_error(error)
        response=dict(success=False,error=code,type=type(error).__name__)
        atomic(root/'diagnostic.json',dict(type=type(error).__name__,traceback=traceback.format_exc(limit=6)))
    atomic(root/'response.json',response)
