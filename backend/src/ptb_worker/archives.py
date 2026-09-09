"""Bounded ZIP IO. Entry names never become filesystem paths; no extractall."""
import io
import stat
import struct
import zipfile
from ptb_api.quota import CHUNK_BYTES, StorageError

MAX_ENTRIES = 16


def reject():
    raise StorageError('archive_rejected',422)


def inspect_zip(reader, size):
    # Bound the central directory BEFORE ZipFile allocates/reads it. ZIP64 and
    # multi-disk archives are outside this initial, deliberately bounded format.
    if size < 22: reject()
    reader.seek(max(0,size-65557))
    tail = reader.read(min(size,65557))
    index = tail.rfind(b'PK\x05\x06')
    if index < 0 or len(tail)-index < 22: reject()
    _,disk,cd_disk,on_disk,total,cd_size,cd_offset,comment = struct.unpack('<4s4H2IH',tail[index:index+22])
    if disk or cd_disk or on_disk!=total or not 0<total<=MAX_ENTRIES or cd_size>131072:
        reject()
    if index+22+comment!=len(tail) or cd_offset+cd_size>size-22-comment:
        reject()
    if b'PK\x06\x07' in tail[max(0,index-20):index]: reject()
    reader.seek(0)


def entry_name(item):
    raw = item.orig_filename
    parts = raw.rstrip('/').split('/')
    mode = item.external_attr >> 16
    if (not raw or len(raw)>180 or '\\' in raw or ':' in raw or raw.startswith('/')
            or len(parts)>8 or any(p in ('','.','..') for p in parts)
            or any(ord(c)<32 for c in raw) or item.flag_bits & 1
            or item.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED)
            or stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR)):
        reject()
    if item.is_dir(): return None
    if item.file_size > max(1,item.compress_size)*200: reject()
    # Preserve a readable logical hierarchy in a flat resource list, without
    # handing an entry path to the OS or accepting colliding display names.
    return ' · '.join(parts)


class InputReader(io.RawIOBase):
    def __init__(self, files, identity, asset):
        self.files,self.identity,self.asset = files,identity,asset
        self.offset = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.offset
    def seek(self,offset,whence=0):
        target = offset if whence==0 else self.offset+offset if whence==1 else self.asset['size_bytes']+offset if whence==2 else -1
        if not 0<=target<=self.asset['size_bytes']: raise OSError('Invalid archive seek')
        self.offset=target
        return target
    def read(self,size=-1):
        count = self.asset['size_bytes']-self.offset if size<0 else min(size,self.asset['size_bytes']-self.offset)
        if count==0:return b''
        if count>CHUNK_BYTES: reject()
        data=self.files.read_input(self.identity,self.asset['id'],self.offset,count)
        self.offset+=len(data)
        return data


class OutputWriter(io.RawIOBase):
    def __init__(self, files, identity, asset_id, stop):
        self.files,self.identity,self.asset_id,self.stop = files,identity,asset_id,stop
        self.offset=0
    def writable(self): return True
    def seekable(self): return False
    def tell(self): return self.offset
    def write(self,data):
        for start in range(0,len(data),CHUNK_BYTES):
            if self.stop.is_set(): raise StorageError('cancelled',409)
            block=data[start:start+CHUNK_BYTES]
            self.files.write(self.identity,self.asset_id,self.offset,block)
            self.offset+=len(block)
        return len(data)


def archive(files,identity,inputs,stop):
    names=[]
    for index,asset in enumerate(inputs):
        # Keep a maximum-length filename intact; ordinary names get an ordinal
        # so common duplicate names stay distinguishable after export.
        base=asset['name']
        name=f'{index+1:02d}-{base}' if len(base)<=177 else base
        entry_name(zipfile.ZipInfo(name))
        names.append(name)
    if len(set(names))!=len(names): reject()
    # Fail before writing for a format we explicitly do not support. Python's
    # conservative writer threshold can be lower than the ZIP format ceiling.
    if any(a['size_bytes']>zipfile.ZIP64_LIMIT for a in inputs): reject()
    if sum(a['size_bytes'] for a in inputs)+sum(128+2*len(n.encode('utf-8')) for n in names)>zipfile.ZIP64_LIMIT: reject()
    output=files.output(identity,'研究文件.zip','archive')
    writer=OutputWriter(files,identity,output['id'],stop)
    with zipfile.ZipFile(writer,'w',compression=zipfile.ZIP_STORED,allowZip64=False) as target:
        for name,asset in zip(names,inputs):
            with target.open(name,'w') as entry:
                offset=0
                while offset<asset['size_bytes']:
                    if stop.is_set(): raise StorageError('cancelled',409)
                    block=files.read_input(identity,asset['id'],offset,min(CHUNK_BYTES,asset['size_bytes']-offset))
                    if not block: reject()
                    entry.write(block);offset+=len(block)
    files.seal(identity,output['id'])


def extract(files,identity,asset,budget,stop):
    reader=InputReader(files,identity,asset)
    inspect_zip(reader,asset['size_bytes'])
    with zipfile.ZipFile(reader,'r',allowZip64=False) as source:
        items=source.infolist()
        if len(items)>MAX_ENTRIES: reject()
        names=[entry_name(item) for item in items]
        real=[name for name in names if name is not None]
        if not real or len(set(real))!=len(real) or sum(item.file_size for item in items)>budget: reject()
        for item,name in zip(items,names):
            if name is None: continue
            output=files.output(identity,name,'archive',item.file_size)
            writer=OutputWriter(files,identity,output['id'],stop)
            with source.open(item,'r') as entry:
                while block:=entry.read(CHUNK_BYTES): writer.write(block)
            if writer.tell()!=item.file_size: reject()
            files.seal(identity,output['id'])
