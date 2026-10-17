"""Bounded transient JSON results. Authentication remains with every caller."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import os
import threading
import time

_scope = ContextVar('preview_owner', default=None)


class PreviewCache:
    def __init__(self, max_bytes=32_000_000, max_entries=24, ttl=60, clock=time.monotonic):
        self.max_bytes,self.max_entries,self.ttl,self.clock=max_bytes,max_entries,ttl,clock
        self.items=OrderedDict();self.bytes=0;self.lock=threading.Lock()

    def _expire(self):
        for key,(deadline,raw) in list(self.items.items()):
            if deadline<=self.clock():self.bytes-=len(raw);del self.items[key]

    def get(self,key):
        with self.lock:
            self._expire();entry=self.items.get(key)
            if entry is None:return None
            self.items.move_to_end(key);return json.loads(entry[1])

    def put(self,key,value):
        raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
        if len(raw)>self.max_bytes:return
        with self.lock:
            self._expire()
            if key in self.items:self.bytes-=len(self.items.pop(key)[1])
            while self.items and (len(self.items)>=self.max_entries or self.bytes+len(raw)>self.max_bytes):
                _,(_,old)=self.items.popitem(last=False);self.bytes-=len(old)
            self.items[key]=(self.clock()+self.ttl,raw);self.bytes+=len(raw)


cache=PreviewCache()


@contextmanager
def preview_scope(owner):
    token=_scope.set(owner)
    try:yield
    finally:_scope.reset(token)


def memoized_preview(kind):
    def decorate(function):
        @wraps(function)
        def call(payload,*args,**kwargs):
            owner=_scope.get()
            if owner is None:
                if os.name!='nt':return function(payload,*args,**kwargs)
                owner=('desktop',os.getpid())
            runtime=None
            if os.name=='posix':
                from .native.linux_runtime import load_profile
                path,_=load_profile();runtime=(str(path),hashlib.sha256(path.read_bytes()).digest())
            key=(owner,runtime,kind,hashlib.sha256(payload).digest(),json.dumps([args,kwargs],sort_keys=True,allow_nan=False))
            value=cache.get(key)
            if value is not None:return value
            value=function(payload,*args,**kwargs)
            cache.put(key,value)
            return value
        return call
    return decorate
