"""M01-C WAV boundary: inspect declared allocation before SciPy decoding.

source_ids: M01-PY-SCIPY, M01-WAVE. No normalization or channel mixing here.
"""
import io
import struct
import math
from uuid import UUID
from scipy.io import wavfile
from phonetic_core.models.audio import AudioInput
from .limits import Limits, LimitedBuffer, LimitError, FormatError


def decode_wav(payload: bytes, limits=Limits()):
    if not isinstance(payload, bytes): raise TypeError('WAV bytes required')
    if len(payload)>limits.input_bytes: raise LimitError('input_bytes_exceeded')
    if len(payload)<12 or payload[:4]!=b'RIFF' or payload[8:12]!=b'WAVE':
        raise FormatError('Only little-endian RIFF WAVE is supported')
    if struct.unpack_from('<I',payload,4)[0]+8!=len(payload):
        raise FormatError('RIFF length mismatch')
    pos=12;fmt=None;data_size=None
    while pos<len(payload):
        if pos+8>len(payload): raise FormatError('Truncated WAV chunk')
        name=payload[pos:pos+4];size=struct.unpack_from('<I',payload,pos+4)[0];pos+=8
        if pos+size>len(payload): raise FormatError('Truncated WAV payload')
        if name==b'fmt ':
            if fmt is not None or size<16 or size>128: raise FormatError('Invalid WAV format chunk')
            fmt=payload[pos:pos+size]
        if name==b'data':
            if data_size is not None: raise FormatError('Multiple WAV data chunks')
            data_size=size
        pos+=size+(size%2)
        # Some SciPy writers omit the final odd-byte padding; no next chunk exists.
        if pos==len(payload)+1 and size%2: pos=len(payload)
    if fmt is None or data_size is None: raise FormatError('Missing WAV chunks')
    tag,channels,rate,byte_rate,block,bits=struct.unpack_from('<HHIIHH',fmt)
    if tag==0xfffe:
        if len(fmt)<40 or fmt[26:40]!=UUID('00000001-0000-0010-8000-00aa00389b71').bytes_le[2:]:
            raise FormatError('Unsupported extensible WAV subtype')
        extra=struct.unpack_from('<H',fmt,16)[0]
        if extra<22 or extra+18>len(fmt):raise FormatError('Invalid extensible WAV size')
        tag=struct.unpack_from('<H',fmt,24)[0]
        if struct.unpack_from('<H',fmt,18)[0] != bits: raise FormatError('Unsupported valid-bit packing')
    if (tag,bits) not in {(1,8),(1,16),(1,24),(1,32),(3,32),(3,64)}:
        raise FormatError('Unsupported WAV sample format')
    if not 0<channels<=limits.channels: raise LimitError('channel_limit_exceeded')
    if rate<=0 or block!=channels*(bits//8) or byte_rate!=rate*block or data_size%block:
        raise FormatError('Invalid WAV sample layout')
    count=data_size//(bits//8)
    if count>limits.samples: raise LimitError('sample_limit_exceeded')
    fs, data=wavfile.read(io.BytesIO(payload))
    if fs!=rate or data.size!=count: raise FormatError('WAV decode mismatch')
    return AudioInput(data,fs)


def encode_wav(audio: AudioInput,limits=Limits()):
    if audio.samples.size>limits.samples:raise LimitError('sample_limit_exceeded')
    if audio.samples.ndim==2 and audio.samples.shape[1]>limits.channels:raise LimitError('channel_limit_exceeded')
    if audio.samples.nbytes+44>limits.output_bytes:raise LimitError('wav_output_limit')
    stream=LimitedBuffer(limits.output_bytes)
    wavfile.write(stream,audio.sample_rate_hz,audio.samples)
    return stream.getvalue()


def slice_wav(audio,start,end,limits=Limits()):
    """Preserve historical int(t*fs) slicing, explicitly refuse invalid ranges."""
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (start,end)):
        raise FormatError('Finite segment times required')
    if not 0<=start<end<=len(audio.samples)/audio.sample_rate_hz:raise FormatError('Invalid segment range')
    first,last=int(start*audio.sample_rate_hz),int(end*audio.sample_rate_hz)
    if first==last:raise FormatError('Empty segment')
    if (last-first)*(audio.samples.shape[1] if audio.samples.ndim==2 else 1)>limits.samples:
        raise LimitError('segment_sample_limit')
    return encode_wav(AudioInput(audio.samples[first:last],audio.sample_rate_hz),limits)
