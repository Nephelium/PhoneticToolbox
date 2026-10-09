"""Task-owned long WAV reader. Scientific rules remain in phonetic_core."""
import hashlib
import io
import struct
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from phonetic_core.egg.bounded import scan, prepare
from .spectrogram_preview import PreviewError

MAX_BYTES = 2_000_000_000


class Recording:
    def __init__(self, path):
        self.path = Path(path)
        if not 0 < self.path.stat().st_size <= MAX_BYTES: raise PreviewError('egg_input_budget')
        self.file = self.path.open('rb')
        try: self._header()
        except BaseException as error:
            self.close()
            if isinstance(error,PreviewError):raise
            if isinstance(error,(struct.error,ValueError,OverflowError)):raise PreviewError('invalid_audio') from None
            raise
        if self.channels != 2: self.close(); raise PreviewError('egg_stereo_required')
        if not 8000 <= self.fs <= 96000: self.close(); raise PreviewError('egg_sample_rate')
        if not 0 < self.frames <= self.fs*1800: self.close(); raise PreviewError('egg_input_budget')
        self.long = self.frames/self.fs > 120 or self.frames > 5_760_000 or self.path.stat().st_size > 64_000_000
        digest = hashlib.sha256()
        with self.path.open('rb') as source:
            for block in iter(lambda:source.read(1048576), b''): digest.update(block)
        self.sha = digest.hexdigest()
        self.reference = None

    def _header(self):
        # Bounded PCM/FLOAT RIFF reader avoids adding libsndfile to the pinned
        # compatibility runtime. Original short files still use SciPy itself.
        header=self.file.read(12)
        if len(header)!=12 or header[:4] not in (b'RIFF',b'RIFX',b'RF64') or header[8:]!=b'WAVE':
            raise PreviewError('invalid_audio')
        self.endian='>' if header[:4]==b'RIFX' else '<';fmt=None;data=None;rf64_size=None
        size=self.path.stat().st_size
        for _ in range(10000):
            chunk=self.file.read(8)
            if len(chunk)!=8:break
            kind=chunk[:4];length=struct.unpack(self.endian+'I',chunk[4:])[0]
            if kind==b'ds64':
                if not 28<=length<=100000:raise PreviewError('invalid_audio')
                raw=self.file.read(length)
                if len(raw)!=length:raise PreviewError('invalid_audio')
                rf64_size=struct.unpack('<Q',raw[8:16])[0]
            elif kind==b'fmt ':
                if not 16<=length<=256:raise PreviewError('invalid_audio')
                fmt=self.file.read(length)
            elif kind==b'data':
                if length==0xffffffff and rf64_size is not None:length=rf64_size
                data=(self.file.tell(),length);self.file.seek(length,1)
            else:self.file.seek(length,1)
            if self.file.tell()>size:raise PreviewError('invalid_audio')
            if length%2:self.file.seek(1,1)
            if fmt is not None and data is not None:break
        if fmt is None or data is None:raise PreviewError('invalid_audio')
        self.format,self.channels,self.fs,_,self.block,self.bits=struct.unpack(self.endian+'HHIIHH',fmt[:16])
        if self.format==65534:
            if len(fmt)<40:raise PreviewError('invalid_audio')
            self.format=struct.unpack(self.endian+'I',fmt[24:28])[0]
        if not ((self.format==1 and self.bits in (8,16,24,32)) or (self.format==3 and self.bits in (32,64))):
            raise PreviewError('invalid_audio')
        if self.channels<1 or self.block!=self.channels*self.bits//8 or data[1]%self.block:
            raise PreviewError('invalid_audio')
        self.offset,self.data_bytes=data;self.frames=data[1]//self.block

    def read(self, first, last):
        if not 0<=first<=last<=self.frames:raise PreviewError('invalid_audio')
        self.file.seek(self.offset+first*self.block)
        raw=self.file.read((last-first)*self.block)
        if len(raw)!=(last-first)*self.block:raise PreviewError('invalid_audio')
        if self.bits==24:
            b=np.frombuffer(raw,dtype=np.uint8).reshape(-1,3).astype(np.int32)
            if self.endian=='>':b=b[:,::-1]
            values=b[:,0]|(b[:,1]<<8)|(b[:,2]<<16)
            values=(values^0x800000)-0x800000
        else:
            dtype=self.endian+('f' if self.format==3 else 'i')+str(self.bits//8)
            if self.bits==8:values=np.frombuffer(raw,dtype=np.uint8).astype(np.int16)-128
            else:values=np.frombuffer(raw,dtype=dtype)
        return values.reshape(last-first,self.channels).astype(np.float32)

    def result(self, config, flip=False):
        if self.reference is None: self.reference = scan(self.read, self.frames, self.fs)
        return prepare(self.read, self.frames, self.fs, self.reference, config, flip_channels=flip)

    def overview(self):
        # Peak-preserving 1 kHz display data, never passed to scientific/IF or
        # playback calculations. Blocks own bins on one global time grid.
        if self.reference is None:self.reference=scan(self.read,self.frames,self.fs)
        rate = 1000; count = int(np.ceil(self.frames*rate/self.fs))
        output = np.empty((count,2), dtype=np.float32)
        for first in range(0,count,20000):
            last = min(count,first+20000)
            edges = np.minimum(self.frames, np.arange(first,last+1,dtype=np.int64)*self.fs//rate)
            data = self.read(int(edges[0]),int(edges[-1]))
            positions = edges[:-1]-edges[0]
            top = np.maximum.reduceat(data,positions,axis=0)
            bottom = np.minimum.reduceat(data,positions,axis=0)
            output[first:last] = np.where(np.abs(top)>=np.abs(bottom),top,bottom)
        output*=np.asarray(.7/np.maximum(self.reference['peaks'],1e-30),dtype=np.float32)
        stream=io.BytesIO();wavfile.write(stream,rate,output)
        return stream.getvalue()

    def close(self):
        if getattr(self,'file',None): self.file.close()
