"""Bounded, host-owned anonymous-RAM checkpoint; never a user file or pickle."""
import json
import os
from pathlib import Path
import re
import stat
import zipfile

import numpy as np

MAX_BYTES=192_000_000
ARRAYS=('time_vector','egg_signal_raw','egg_signal_processed','audio_signal','gci_times','goi_times','peak_times',
        'cq_times','cq_values','sq_values','gci_f0_times','gci_f0_values','audio_f0_times','audio_f0_values','reaper_f0_times','reaper_f0_values')
LISTS={'gci_times','goi_times','peak_times'}


def _open(path,mode,kind='state'):
    if not isinstance(path,str) or not re.fullmatch(r'/proc/[1-9][0-9]*/fd/[0-9]+',path):raise ValueError('Not a host memory capability')
    if not os.readlink(path).startswith('/memfd:ptb-egg-'+kind+'-'):raise ValueError('Not an EGG memory capability')
    stream=open(path,mode);info=os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_size>MAX_BYTES:
        stream.close();raise ValueError('Invalid memory checkpoint')
    return stream


def source_bytes(path,size):
    import fcntl
    from .native.anonymous_memory import GET_SEALS, IMMUTABLE
    with _open(path,'rb','source') as stream:
        if os.fstat(stream.fileno()).st_size!=size:raise ValueError('Invalid source size')
        if fcntl.fcntl(stream,GET_SEALS)&IMMUTABLE!=IMMUTABLE:raise ValueError('Mutable source memory')
        raw=stream.read(size)
        if len(raw)!=size:raise ValueError('Incomplete source memory')
        return raw


def restore(session,path):
    if not path:return False
    try:
        with _open(path,'rb') as stream:
            if not os.fstat(stream.fileno()).st_size:return False
            with zipfile.ZipFile(stream) as archive:
                total=0
                for item in archive.infolist():
                    if item.filename not in {n+'.npy' for n in (*ARRAYS,'meta','main_spectral','main_png')}:raise ValueError()
                    total+=item.file_size
                    if total>MAX_BYTES:raise ValueError()
                    with archive.open(item) as part:
                        version=np.lib.format.read_magic(part)
                        shape,order,dtype=np.lib.format._read_array_header(part,version,max_header_size=10000)
                        if dtype.hasobject or len(shape)>2 or any(n<0 for n in shape) or int(np.prod(shape,dtype=object))*dtype.itemsize>item.file_size:raise ValueError()
            stream.seek(0)
            with np.load(stream,allow_pickle=False) as data:
                encoded=data['meta'];assert encoded.dtype==np.uint8 and encoded.ndim==1 and encoded.size<=8_000_000
                meta=json.loads(encoded.tobytes())
                if meta['source']!=session.sha or meta['fs']!=session.fs or meta['samples']!=len(session.samples):return False
                from phonetic_core.egg import EGGConfig
                from phonetic_core.egg.model import EGGAnalysisResult
                values={k:(data[k].tolist() if k in LISTS else data[k]) for k in meta['arrays']}
                for k in ('preprocessing_config','analysis_config'):
                    values[k]=EGGConfig(**meta[k]) if meta[k] is not None else None
                values.update(meta['scalars']);values['source_ids']=tuple(values['source_ids'])
                session.result=EGGAnalysisResult(**values)
                session.prepared_key=tuple(meta['prepared_key']) if meta['prepared_key'] is not None else None
                session.events_key=tuple(meta['events_key']) if meta['events_key'] is not None else None
                session.pitch_bounds=tuple(meta['pitch_bounds']) if meta['pitch_bounds'] is not None else None
                session.main.clear()
                if meta['main'] is not None:
                    key,main=meta['main'];key=(key[0],key[1],tuple(tuple(p) for p in key[2]))
                    main['_spectral_db']=data['main_spectral'] if 'main_spectral' in data else None
                    main['png']=data['main_png'].tobytes()
                    for name in ('_display_range','spectral_extent','spectral_shape','raster_shape','suggested_db_range'):
                        if main.get(name) is not None:main[name]=tuple(main[name])
                    session.main[key]=main
        return True
    except (OSError,ValueError,KeyError,TypeError,AssertionError,zipfile.BadZipFile,EOFError):
        session.result=None;session.prepared_key=session.events_key=session.pitch_bounds=None;session.main.clear()
        return False


def save(session,path):
    if not path or session.result is None:return False
    from dataclasses import asdict
    result=session.result;arrays={k:np.asarray(getattr(result,k)) for k in ARRAYS if getattr(result,k) is not None}
    if sum(a.nbytes for a in arrays.values())>MAX_BYTES-8_100_000:return False
    meta=dict(source=session.sha,fs=int(session.fs),samples=len(session.samples),arrays=list(arrays),prepared_key=session.prepared_key,
              events_key=session.events_key,pitch_bounds=session.pitch_bounds,scalars={k:getattr(result,k) for k in ('fs','file_duration','method_version','source_ids','glottal_movement_events')},
              preprocessing_config=asdict(result.preprocessing_config) if result.preprocessing_config is not None else None,
              analysis_config=asdict(result.analysis_config) if result.analysis_config is not None else None,main=None)
    if session.main:
        from ptb_api.egg_models import EggSeries
        key,main=next(iter(session.main.items()));meta['main']=(key,{k:(v.model_dump() if isinstance(v,EggSeries) else v) for k,v in main.items() if k not in ('png','_spectral_db')})
        arrays['main_png']=np.frombuffer(main['png'],dtype='uint8')
        if main.get('_spectral_db') is not None:arrays['main_spectral']=main['_spectral_db']
    try:encoded=json.dumps(meta,allow_nan=False,separators=(',',':')).encode()
    except (TypeError,ValueError,OverflowError):return False
    if len(encoded)>8_000_000 or sum(a.nbytes for a in arrays.values())+len(encoded)>MAX_BYTES-100_000:return False
    arrays['meta']=np.frombuffer(encoded,dtype='uint8')
    try:
        with _open(path,'r+b') as stream:
            stream.seek(0);stream.truncate(0)
            np.savez(stream,**arrays)
            if stream.tell()>MAX_BYTES:stream.truncate(0);return False
        return True
    except (OSError,ValueError):return False
