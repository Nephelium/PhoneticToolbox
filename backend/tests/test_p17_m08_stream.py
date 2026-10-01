"""P17 framing checks use arbitrary bytes, never synthesized audio."""
import hashlib,json,struct
import pytest
from ptb_worker.m08_stream import Receiver
from ptb_worker.io.limits import FormatError
class Files:
 def __init__(self):self.raw={};self.sizes=[];self.sealed=[]
 def output(self,identity,name,kind,size):self.raw[name]=bytearray();return {'id':name}
 def write(self,identity,id,offset,raw):assert offset==len(self.raw[id]);self.raw[id].extend(raw);self.sizes.append(len(raw))
 def seal(self,identity,id):self.sealed.append(id)
def frame(header,raw=b''):
 data=json.dumps(header).encode();return struct.pack('<I',len(data))+data+raw
def stream(raw,sha=None):
 return frame(dict(kind='file',name='m08.ptb.json',size=len(raw),sha256=sha or hashlib.sha256(raw).hexdigest()),raw)+frame(dict(kind='complete',audio_sha256='source',count=1))
@pytest.mark.parametrize('chunk',[1,4096,65536,1048576])
def test_fragmented_transport_preserves_bytes_and_bounds(chunk):
 raw=bytes(range(251))*8400;files=Files();receiver=Receiver(files,('test',), 'source',write_chunk_bytes=1048576);data=stream(raw)
 # one-byte splits cover header edges without millions of Python calls.
 step=chunk if chunk>1 else 997
 for i in range(0,len(data),step):receiver.write(data[i:i+step])
 receiver.finish();assert bytes(files.raw['m08.ptb.json'])==raw;assert len(files.sizes)==3;assert max(files.sizes)<=1048576;assert len(files.sealed)==1
@pytest.mark.parametrize('case',['hash','truncated','trailing'])
def test_invalid_stream_never_finishes(case):
 raw=b'framing payload';data=stream(raw,'0'*64 if case=='hash' else None)
 if case=='truncated':data=data[:-4]
 if case=='trailing':data+=b'extra'
 receiver=Receiver(Files(),('test',),'source',write_chunk_bytes=1048576)
 with pytest.raises(FormatError):receiver.write(data);receiver.finish()
@pytest.mark.parametrize('size',[0,1048577,True])
def test_invalid_write_bound(size):
 with pytest.raises(ValueError):Receiver(Files(),('test',),'source',write_chunk_bytes=size)
