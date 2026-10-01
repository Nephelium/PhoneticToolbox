"""Scratch capabilities backed by pre-reserved P07/local assets, never loose files."""
import os
from pathlib import Path
from .io.scratch import Scratch,no_links
from .io.limits import LimitError


class ManagedScratch(Scratch):
    def __init__(self,files,identity,budget=128_000_000,*,write_chunk_bytes=65536):
        if type(write_chunk_bytes) is not int or not 0 < write_chunk_bytes <= 1_048_576:
            raise ValueError('Invalid scratch write chunk size')
        self.write_chunk_bytes=write_chunk_bytes
        self.adapter,self.identity=files,identity
        self.budget,self.used=budget,0;self.files={};self.ids={}
        self.root=files.scratch_root

    def create(self,data,suffix):
        if suffix not in ('.json','.wav','.xlsx','.sqlite'):raise ValueError('Unsupported scratch type')
        if self.used+len(data)>self.budget:raise LimitError('scratch_budget_exceeded')
        asset=self.adapter.output(self.identity,'worker-scratch'+suffix,'temporary',len(data))
        path=self.adapter.scratch_path(self.identity,asset['id'])
        self.files[path]=len(data);self.ids[path]=asset['id'];self.used+=len(data)
        try:
            for offset in range(0,len(data),self.write_chunk_bytes):
                self.adapter.write(self.identity,asset['id'],offset,data[offset:offset+self.write_chunk_bytes])
        except BaseException:self.remove(path);raise
        return path

    def remove(self,path):
        if path not in self.files:raise ValueError('Not an owned scratch resource')
        self.adapter.release_scratch(self.identity,self.ids[path])
        self.used-=self.files.pop(path);self.ids.pop(path)

    def close(self):
        for path in list(self.files):self.remove(path)


class ReservedNativeScratch(Scratch):
    """Trusted child may write only one newly preallocated, quota-covered WAV slot."""
    def __init__(self,path,budget):
        self.path=Path(path);no_links(self.path)
        self.root=self.path.parent;self.budget=budget;self.used=0;self.files={}
        if not self.path.is_file() or self.path.stat().st_size:raise ValueError('Native scratch must be empty')

    def create(self,data,suffix):
        if suffix!='.wav' or self.used or len(data)>self.budget:raise LimitError('native_scratch_budget')
        no_links(self.path)
        with self.path.open('r+b') as handle:
            if os.fstat(handle.fileno()).st_size:raise ValueError('Native scratch changed')
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        self.used=len(data);self.files[self.path]=len(data);return self.path

    def remove(self,path):
        if path!=self.path or path not in self.files:raise ValueError('Unowned native scratch')
        no_links(path)
        with path.open('r+b') as handle:handle.truncate(0);handle.flush();os.fsync(handle.fileno())
        self.files.clear();self.used=0

    def close(self):
        if self.files:self.remove(self.path)
