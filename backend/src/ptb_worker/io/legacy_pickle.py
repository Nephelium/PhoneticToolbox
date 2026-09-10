"""M01-PICKLE: symbolic pickle reader, never imports/calls serialized globals.

Only primitive containers and numeric NumPy 1/2 scalar/ndarray encodings used by
the v2 lip writer are interpreted. Numeric arrays remain bounded byte records;
unused landmarks are validated but never expanded into Python float objects.
"""
from dataclasses import dataclass
import math
import pickletools
import re
import struct
import sys
from .limits import FormatError, LimitError


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass
class Dtype:
    code: str
    endian: str = '<' if sys.byteorder=='little' else '>'
    built: bool = False

    @property
    def format(self):
        return self.endian+{'f2':'e','f4':'f','f8':'d','i1':'b','u1':'B','i2':'h',
            'u2':'H','i4':'i','u4':'I','i8':'q','u8':'Q','b1':'?'}[self.code]


@dataclass
class Array:
    shape: tuple | None = None
    dtype: Dtype | None = None
    raw: bytes = b''

    def vector(self):
        if self.shape is None or len(self.shape)!=1:raise FormatError('Legacy vector must be one dimensional')
        return [v[0] for v in struct.iter_unpack(self.dtype.format,self.raw)]


def read_numeric_pickle(payload,limits):
    if type(payload)!=bytes or not 0<len(payload)<=limits.input_bytes:raise LimitError('legacy_pickle_input_budget')
    stack=[];marks=[];memo={};nodes=0;array_values=0
    def symbol(module,name):
        if type(module)!=str or type(name)!=str:raise FormatError('Invalid symbolic global')
        if (module,name) in [('numpy','dtype'),('numpy','ndarray'),('_codecs','encode')]:return Symbol(name)
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('scalar','_reconstruct'):return Symbol(name)
        if module in ('numpy.core.numeric','numpy._core.numeric') and name=='_frombuffer':return Symbol(name)
        raise FormatError('Unsupported legacy global')
    def array(shape,dtype,raw):
        nonlocal array_values
        if type(shape)!=tuple or not 1<=len(shape)<=4 or any(type(n)!=int or n<0 for n in shape):raise FormatError('Invalid array shape')
        count=math.prod(shape)
        if count>limits.samples or array_values+count>limits.samples:raise LimitError('legacy_array_budget')
        if type(dtype)!=Dtype or type(raw)!=bytes or len(raw)!=count*struct.calcsize(dtype.format):raise FormatError('Invalid numeric array')
        array_values+=count
        return Array(shape,Dtype(dtype.code,dtype.endian,True),raw)
    def reduce(value,args):
        if type(value)!=Symbol or type(args)!=tuple:raise FormatError('Unsupported legacy reduce')
        if value.name=='dtype':
            if len(args)!=3 or args[1:]!=(False,True) or type(args[0])!=str or not re.fullmatch(r'[<>=|]?(f[248]|[iu][1248]|b1)',args[0]):raise FormatError('Unsupported numeric dtype')
            code=args[0];endian=code[0] if code[0] in '<>' else ('<' if sys.byteorder=='little' else '>')
            return Dtype(code.lstrip('<>=|'),endian)
        if value.name=='scalar':
            if len(args)!=2 or type(args[0])!=Dtype or type(args[1])!=bytes or len(args[1])!=struct.calcsize(args[0].format):raise FormatError('Invalid numeric scalar')
            return struct.unpack(args[0].format,args[1])[0]
        if value.name=='_reconstruct':
            if args!=(Symbol('ndarray'),(0,),b'b'):raise FormatError('Unsupported array reconstruction')
            return Array()
        if value.name=='_frombuffer':
            if len(args)!=4 or args[3] not in ('C','F'):raise FormatError('Unsupported array layout')
            return array(args[2],args[1],args[0])
        if value.name=='encode' and len(args)==2 and type(args[0])==str and args[1]=='latin1':return args[0].encode('latin1')
        raise FormatError('Unsupported legacy reduce')
    def items():
        if not marks:raise FormatError('Unmatched legacy mark')
        start=marks.pop();values=stack[start:];del stack[start:];return values
    def mapping(target,values):
        if type(target)!=dict or len(values)%2:raise FormatError('Invalid legacy mapping')
        for key,value in zip(values[::2],values[1::2]):
            if type(key)!=str or key in target:raise FormatError('Duplicate or non-text legacy key')
            target[key]=value
    try:
        for opcode,arg,pos in pickletools.genops(payload):
            nodes+=1
            if nodes>limits.text_items*8 or len(stack)>limits.text_items or len(memo)>limits.text_items or len(marks)>16:raise LimitError('legacy_pickle_structure_budget')
            op=opcode.name
            if op=='PROTO':
                if not 0<=arg<=5:raise FormatError('Unsupported pickle protocol')
            elif op=='FRAME':
                if arg>len(payload)-pos-9:raise FormatError('Invalid pickle frame')
            elif op=='MARK':marks.append(len(stack))
            elif op in ('NONE','NEWTRUE','NEWFALSE'):stack.append({'NONE':None,'NEWTRUE':True,'NEWFALSE':False}[op])
            elif op in ('INT','BININT','BININT1','BININT2','LONG','LONG1','LONG4','FLOAT','BINFLOAT','STRING','UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','BINSTRING','SHORT_BINSTRING','BINBYTES','SHORT_BINBYTES','BINBYTES8','BYTEARRAY8'):
                stack.append(bytes(arg) if op=='BYTEARRAY8' else arg)
            elif op in ('EMPTY_LIST','EMPTY_TUPLE','EMPTY_DICT'):stack.append({'EMPTY_LIST':list,'EMPTY_TUPLE':tuple,'EMPTY_DICT':dict}[op]())
            elif op in ('LIST','TUPLE'):stack.append(items() if op=='LIST' else tuple(items()))
            elif op=='DICT':
                values=items();target={};mapping(target,values);stack.append(target)
            elif op in ('TUPLE1','TUPLE2','TUPLE3'):
                n=int(op[-1]);values=stack[-n:];del stack[-n:];stack.append(tuple(values))
            elif op in ('APPEND','APPENDS'):
                values=[stack.pop()] if op=='APPEND' else items()
                if type(stack[-1])!=list:raise FormatError('Invalid legacy list')
                stack[-1].extend(values)
                if len(stack[-1])>limits.text_items:raise LimitError('legacy_list_budget')
            elif op in ('SETITEM','SETITEMS'):
                values=items() if op=='SETITEMS' else stack[-2:]
                if op=='SETITEM':del stack[-2:]
                mapping(stack[-1],values)
            elif op in ('PUT','BINPUT','LONG_BINPUT','MEMOIZE'):
                key=len(memo) if op=='MEMOIZE' else int(arg)
                if not 0<=key<limits.text_items:raise LimitError('legacy_memo_budget')
                if key in memo:raise FormatError('Repeated legacy memo')
                memo[key]=stack[-1]
            elif op in ('GET','BINGET','LONG_BINGET'):stack.append(memo[int(arg)])
            elif op=='GLOBAL':stack.append(symbol(*arg.split(' ')))
            elif op=='STACK_GLOBAL':
                name=stack.pop();module=stack.pop();stack.append(symbol(module,name))
            elif op=='REDUCE':
                args=stack.pop();value=stack.pop();stack.append(reduce(value,args))
            elif op=='BUILD':
                state=stack.pop();value=stack[-1]
                if type(value)==Dtype:
                    if value.built or type(state)!=tuple or len(state)!=8 or state[0]!=3 or state[1] not in '<>=|' or state[2:]!=(None,None,None,-1,-1,0):raise FormatError('Unsupported dtype state')
                    if state[1] in '<>':value.endian=state[1]
                    value.built=True
                elif type(value)==Array:
                    if value.shape is not None or type(state)!=tuple or len(state)!=5 or state[0]!=1 or type(state[3])!=bool:raise FormatError('Unsupported array state')
                    parsed=array(state[1],state[2],state[4]);value.shape,value.dtype,value.raw=parsed.shape,parsed.dtype,parsed.raw
                else:raise FormatError('Unsupported legacy state')
            elif op=='STOP':
                if marks or len(stack)!=1 or pos+1!=len(payload):raise FormatError('Incomplete or trailing pickle')
                result=stack[0];break
            else:raise FormatError('Unsupported legacy opcode')
        else:raise FormatError('Incomplete pickle')
        seen=set();count=0
        def check(value,depth=0):
            nonlocal count
            count+=1
            if depth>16 or count>limits.text_items:raise LimitError('legacy_tree_budget')
            if type(value) in (list,tuple,dict,Array):
                if id(value) in seen:raise FormatError('Shared or cyclic legacy containers')
                seen.add(id(value))
                if type(value)==Array:
                    if value.shape is None:raise FormatError('Incomplete array')
                else:
                    for child in (value.values() if type(value)==dict else value):check(child,depth+1)
            elif type(value) not in (str,int,float,bool,type(None)):raise FormatError('Unsupported legacy value')
        check(result)
        return result
    except (FormatError,LimitError):raise
    except (ValueError,TypeError,IndexError,KeyError,OverflowError,struct.error):raise FormatError('Invalid numeric legacy pickle') from None
