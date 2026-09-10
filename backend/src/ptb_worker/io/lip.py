"""M01-C source_id M01-PICKLE: inert JSON and local-only legacy conversion.

The local conversion route uses a separate bounded child and symbolic numeric
reader. No executable pickle loader is exposed. The older primitive-only helper
remains restricted; wire JSON never contains Python objects or constructors.
"""
import io
import json
import math
import pickle
import pickletools
from .limits import Limits, LimitError, FormatError

VECTORS=('absolute_timestamps','relative_times','area','outer_width','open','circularity')
METADATA=('audio_first_frame_time','lip_manual_offset','time_alignment_mode')


def bounded_tree(value,limits):
    stack=[(value,0)];seen=set();count=0
    while stack:
        node,depth=stack.pop();count+=1
        if count>limits.text_items or depth>16: raise LimitError('lip_structure_limit')
        if type(node) in (dict,list,tuple):
            if id(node) in seen: raise FormatError('Shared or cyclic lip containers')
            seen.add(id(node))
            if len(node)>limits.text_items-count: raise LimitError('lip_item_limit')
            if type(node)==dict:
                if any(type(k)!=str for k in node): raise FormatError('String lip keys required')
                stack.extend((v,depth+1) for v in node.values())
            else:stack.extend((v,depth+1) for v in node)
        elif type(node) not in (str,int,float,bool,type(None)):
            raise FormatError('Unsupported lip value type')
        elif type(node)==str and len(node)>limits.text_bytes: raise LimitError('lip_string_limit')


def normalize(data,limits,*,wire=False):
    bounded_tree(data,limits)
    if type(data)!=dict: raise FormatError('Lip object required')
    result={};total=0
    for key in VECTORS:
        if key not in data:continue
        values=data[key]
        if wire:
            if type(values)!=dict or set(values)!={'values','nonfinite'}:raise FormatError('Masked lip vector required')
            raw,masks=values['values'],values['nonfinite']
            if type(raw)!=list or type(masks)!=list or len(raw)!=len(masks):raise FormatError('Invalid lip masks')
            values=[]
            for value,mask in zip(raw,masks):
                if type(mask)!=int or mask not in (0,1,2,3):raise FormatError('Invalid lip nonfinite code')
                if mask:
                    if value is not None:raise FormatError('Nonfinite lip value must be null')
                    value=(float('nan'),float('inf'),float('-inf'))[mask-1]
                elif type(value) not in (int,float) or not math.isfinite(value):raise FormatError('Invalid finite lip value')
                values.append(value)
        if type(values) not in (list,tuple):raise FormatError('Lip vector required')
        total+=len(values)
        if total>limits.samples:raise LimitError('lip_sample_limit')
        out=[]
        for value in values:
            if type(value) not in (float,int):raise FormatError('Numeric lip samples required')
            try:out.append(float(value))
            except OverflowError as exc:raise FormatError('Excessive lip number') from exc
        result[key]=out
    if not any(result.get(k) for k in VECTORS[:2]):raise FormatError('Lip time axis required')
    meta=data.get('metadata',{})
    if type(meta)!=dict:raise FormatError('Lip metadata object required')
    clean={}
    for key in METADATA:
        if key not in meta:continue
        value=meta[key]
        if key=='time_alignment_mode':
            if type(value)!=str or len(value)>128:raise FormatError('Invalid lip alignment mode')
        elif value is not None:
            if type(value) not in (float,int):raise FormatError('Invalid lip anchor or offset')
            value=float(value)
            if not math.isfinite(value):
                if key=='lip_manual_offset':raise FormatError('Nonfinite lip offset')
                value=None
        clean[key]=value
    result['metadata']=clean
    return result


def encode_lip(data,limits=Limits()):
    clean=normalize(data,limits)
    for key in VECTORS:
        if key in clean:
            vector=clean[key]
            clean[key]={'values':[v if math.isfinite(v) else None for v in vector],
                        'nonfinite':[0 if math.isfinite(v) else 1 if math.isnan(v) else 2 if v>0 else 3 for v in vector]}
    text=json.dumps({'schema':'ptb.lip/1','data':clean},ensure_ascii=False,allow_nan=False,separators=(',',':'))
    payload=text.encode('utf-8')
    if len(payload)>min(limits.input_bytes,limits.text_bytes):raise LimitError('lip_bytes_exceeded')
    return payload


def decode_lip(payload,limits=Limits()):
    if not isinstance(payload,bytes):raise TypeError('Lip JSON bytes required')
    if len(payload)>min(limits.input_bytes,limits.text_bytes):raise LimitError('lip_bytes_exceeded')
    depth=0;quoted=False;escaped=False;items=0
    for byte in payload:
        if quoted:
            if escaped:escaped=False
            elif byte==92:escaped=True
            elif byte==34:quoted=False
        elif byte==34:quoted=True
        elif byte in (91,123):
            depth+=1;items+=1
            if depth>16:raise LimitError('lip_structure_limit')
        elif byte in (93,125):depth-=1
        elif byte in (44,58):items+=1
        if items>limits.text_items*2:raise LimitError('lip_item_limit')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise FormatError('Duplicate lip JSON key')
            d[k]=v
        return d
    try:
        obj=json.loads(payload,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(FormatError('Nonstandard JSON number')))
    except (ValueError,RecursionError,UnicodeError) as exc:raise FormatError('Invalid lip JSON') from exc
    bounded_tree(obj,limits)
    if type(obj)!=dict or set(obj)!={'schema','data'} or obj['schema']!='ptb.lip/1':raise FormatError('Unsupported lip schema')
    return normalize(obj['data'],limits,wire=True)


class InertUnpickler(pickle.Unpickler):
    def find_class(self,*args):raise FormatError('Pickle globals are forbidden')
    def persistent_load(self,*args):raise FormatError('Pickle persistent IDs are forbidden')


def load_inert_legacy_pickle(payload,limits=Limits()):
    if not isinstance(payload,bytes):raise TypeError('Legacy pickle bytes required')
    if len(payload)>min(limits.input_bytes,limits.text_bytes):raise LimitError('pickle_bytes_exceeded')
    forbidden={'GLOBAL','STACK_GLOBAL','REDUCE','BUILD','NEWOBJ','NEWOBJ_EX','INST','OBJ','EXT1','EXT2','EXT4','PERSID','BINPERSID','NEXT_BUFFER','READONLY_BUFFER'}
    try:
        end=None
        for count,(op,arg,pos) in enumerate(pickletools.genops(payload),1):
            if count>limits.text_items*4:raise LimitError('pickle_opcode_limit')
            if op.name in forbidden:raise FormatError('Executable pickle opcode: '+op.name)
            if op.name in ('PUT','BINPUT','LONG_BINPUT','GET','BINGET','LONG_BINGET') and arg>=limits.text_items:
                raise LimitError('pickle_memo_limit')
            if op.name=='FRAME' and arg>len(payload)-pos-9:raise FormatError('Truncated pickle frame')
            if op.name=='STOP':end=pos+1
        if end!=len(payload):raise FormatError('Trailing or truncated pickle data')
        value=InertUnpickler(io.BytesIO(payload)).load()
    except (pickle.UnpicklingError,ValueError,EOFError,RecursionError,OverflowError) as exc:
        if isinstance(exc,(LimitError,FormatError)):raise
        raise FormatError('Invalid inert pickle') from exc
    bounded_tree(value,limits)
    return value


def convert_local_legacy_lip(payload,limits=Limits(),*,companion=None):
    from .legacy_pickle import read_numeric_pickle,Array
    data=read_numeric_pickle(payload,limits)
    if type(data)!=dict:raise FormatError('Legacy lip mapping required')
    selected={k:(data[k].vector() if type(data[k])==Array else data[k]) for k in VECTORS if k in data}
    metadata=data.get('metadata',{})
    if type(metadata)!=dict:raise FormatError('Invalid legacy metadata')
    selected['metadata']={k:metadata[k] for k in METADATA if k in metadata}
    if selected['metadata'].get('audio_first_frame_time') is None and companion is not None:
        info=read_numeric_pickle(companion,limits)
        start=info.get('start_time') if type(info)==dict else None
        if type(start) not in (int,float) or not math.isfinite(start):raise FormatError('Invalid companion audio start')
        selected['metadata']['audio_first_frame_time']=start
    return encode_lip(selected,limits)


def legacy_companion_start(payload,limits=Limits()):
    value=load_inert_legacy_pickle(payload,limits)
    if type(value)!=dict or type(value.get('start_time')) not in (int,float):raise FormatError('Invalid companion start')
    start=float(value['start_time'])
    if not math.isfinite(start):raise FormatError('Nonfinite companion start')
    return start
