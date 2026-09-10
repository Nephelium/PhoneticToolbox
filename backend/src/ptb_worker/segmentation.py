"""M01-F1 prepare a complete single-audio segment bundle in an owned child.

Not yet a public job or writer: callers must add durable owner/fencing/quota and
atomic publication. Host Scratch owns only the new temporary request bytes.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
import struct
import sys
import time
from .io.limits import Limits,LimitError,FormatError,Cancelled
from .io.scratch import Scratch
from .acoustic_errors import ACOUSTIC_ERRORS,AcousticFailure

SEGMENT_LIMITS=Limits(input_bytes=64_000_000,samples=32_000_000,channels=32,
                     output_bytes=64_000_000,process_bytes=1_000_000_000,timeout_seconds=60.)
MAX_MANIFEST_BYTES=2_000_000


@dataclass(frozen=True)
class SegmentBundle:
    manifest: dict
    payloads: tuple[bytes,...]


def digest(raw):return hashlib.sha256(raw).hexdigest()


def unpack_bundle(payload,limit):
    if len(payload)>limit or len(payload)<8:raise FormatError('invalid_segment_bundle')
    n=struct.unpack_from('<Q',payload)[0]
    if not 0<n<=MAX_MANIFEST_BYTES or 8+n>len(payload):raise FormatError('invalid_segment_manifest')
    try:manifest=json.loads(payload[8:8+n])
    except (ValueError,UnicodeError):raise FormatError('invalid_segment_manifest') from None
    if not isinstance(manifest,dict):raise FormatError('invalid_segment_manifest')
    offset=8+n;blobs=[]
    error=manifest.get('error')
    if isinstance(error,str) and error in ACOUSTIC_ERRORS:
        raise AcousticFailure(error)
    if manifest.get('kind') not in ('prepared_segments','prepared_analysis') or not isinstance(manifest.get('files'),list) or not 1<=len(manifest['files'])<=3000:
        raise FormatError('invalid_segment_manifest')
    names=set()
    for entry in manifest['files']:
        if not isinstance(entry,dict):raise FormatError('invalid_segment_file')
        name,size=entry.get('name'),entry.get('size_bytes')
        kind=entry.get('format')
        extension={'wav':'.wav','xlsx':'.xlsx','sqlite':'.ptb.sqlite','json':'.ptb.json'}.get(kind) if isinstance(kind,str) else None
        if (type(size)!=int or size<=0 or offset+size>len(payload) or not isinstance(name,str) or
            not extension or not name.endswith(extension) or len(name)>220 or
            any(ord(c)<32 or c in '/\\:<>"|?*' for c in name) or name.casefold() in names):
            raise FormatError('invalid_segment_file')
        raw=payload[offset:offset+size]
        if digest(raw)!=entry.get('sha256'):raise FormatError('segment_digest_mismatch')
        names.add(name.casefold());blobs.append(raw);offset+=size
    if offset!=len(payload):raise FormatError('trailing_segment_bytes')
    return SegmentBundle(manifest,tuple(blobs))


def prepare_segments(audio,textgrid,layer,scratch,*,audio_name='audio.wav',parent_result=None,
                     legacy_result=None,legacy_name=None,limits=SEGMENT_LIMITS,stop=lambda:False,on_started=None,heartbeat=lambda:None):
    """Return verified bytes, never publish paths. Heartbeat runs during child work.

    parent_result requires authenticated matching-WAV provenance. legacy_result
    is a separately identified, explicitly associated historical table.
    """
    if not isinstance(scratch,Scratch):raise TypeError('Host-owned Scratch required')
    if stop():raise Cancelled('cancelled')
    if type(audio)!=bytes or type(textgrid)!=bytes:raise TypeError('Audio and TextGrid bytes required')
    if len(audio)>limits.input_bytes or len(textgrid)>limits.text_bytes:raise LimitError('segment_input_budget')
    if not isinstance(layer,str) or not 0<len(layer)<=180:raise FormatError('invalid_segment_layer')
    if not isinstance(audio_name,str) or not 0<len(audio_name)<=255:raise FormatError('invalid_audio_name')
    if legacy_result is not None:
        from .io.legacy_parameters import SUFFIXES
        if parent_result is not None or type(legacy_name)!=str or not 0<len(legacy_name)<=255 or not legacy_name.lower().endswith(SUFFIXES):raise FormatError('invalid_legacy_source')
        parent_result=legacy_result
    if parent_result is not None and (type(parent_result)!=bytes or len(parent_result)>16_000_000):raise LimitError('parent_result_budget')
    header={'audio_size':len(audio),'grid_size':len(textgrid),'parent_size':len(parent_result or b''),
            'audio_sha256':digest(audio),'textgrid_sha256':digest(textgrid),'layer':layer,
            'audio_name':audio_name,'legacy_name':legacy_name if legacy_result is not None else None,'limits':asdict(limits)}
    raw=json.dumps(header,ensure_ascii=False,separators=(',',':')).encode()+b'\n'+audio+textgrid+(parent_result or b'')
    path=scratch.create(raw,'.json')
    try:
        from .native.windows import InputPipe
        from .native.reaper import collect_pipe
        last_beat=-float('inf')
        def check():
            nonlocal last_beat
            if time.monotonic()-last_beat>=.5:
                heartbeat()  # A lease failure must abort, not be converted to success.
                last_beat=time.monotonic()
            return stop()
        pipe=InputPipe()
        payload,_=collect_pipe([sys.executable,'-B','-m','ptb_worker.segment_child',str(path),pipe.name],
                               pipe,scratch.root,limits,check,on_started)
        if stop():raise Cancelled('cancelled')
        heartbeat()
        bundle=unpack_bundle(payload,limits.output_bytes)
        if bundle.manifest['audio_sha256']!=header['audio_sha256'] or bundle.manifest['textgrid_sha256']!=header['textgrid_sha256']:
            raise FormatError('segment_source_mismatch')
        return bundle
    finally:scratch.remove(path)
