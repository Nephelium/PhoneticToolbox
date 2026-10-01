"""Bounded scratch writes retain every byte, reservation and failure cleanup."""
import pytest
from ptb_worker.managed_scratch import ManagedScratch


class Files:
    scratch_root = None
    def __init__(self, root): self.scratch_root=root; self.writes=[]; self.released=[]
    def output(self, identity, name, kind, size): self.reserved=size; return {'id':'owned'}
    def scratch_path(self, identity, asset): return self.scratch_root/'owned.bin'
    def write(self, identity, asset, offset, data): self.writes.append((offset,data))
    def release_scratch(self, identity, asset): self.released.append(asset)


@pytest.mark.parametrize('chunk',[65536,262144,1048576])
def test_configured_write_chunks_preserve_payload_and_owned_cleanup(tmp_path,chunk):
    files=Files(tmp_path);data=b'abc' * 700001
    with ManagedScratch(files,('job','worker',1),write_chunk_bytes=chunk) as scratch:
        path=scratch.create(data,'.json')
        assert path==tmp_path/'owned.bin' and files.reserved==len(data)
        assert b''.join(raw for _,raw in files.writes)==data
        assert [offset for offset,_ in files.writes]==list(range(0,len(data),chunk))
        assert all(len(raw)<=chunk for _,raw in files.writes)
    assert files.released==['owned']


def test_failed_large_write_releases_only_its_reservation(tmp_path):
    files=Files(tmp_path)
    def fail(*args): raise OSError('write failed')
    files.write=fail
    with ManagedScratch(files,('job','worker',1),write_chunk_bytes=1048576) as scratch:
        with pytest.raises(OSError,match='write failed'):scratch.create(b'a'*1048577,'.json')
        assert scratch.used==0
    assert files.released==['owned']
