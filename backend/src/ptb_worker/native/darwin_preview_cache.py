"""Transient, bounded results for local Mac previews. No audio or disk cache."""
from collections import OrderedDict
import json
import threading
import time


class ResultCache:
    def __init__(self,max_bytes=32_000_000,max_entries=8,ttl=60):
        self.max_bytes=max_bytes;self.max_entries=max_entries;self.ttl=ttl
        self.condition=threading.Condition();self.values=OrderedDict();self.pending=set()
        self.bytes=0;self.hits=0;self.misses=0

    def get(self,key,compute):
        with self.condition:
            while key in self.pending:self.condition.wait()
            now=time.monotonic()
            for old,(_,expiry) in list(self.values.items()):
                if expiry<=now:self._remove(old)
            if key in self.values:
                raw,_=self.values[key];self.values.move_to_end(key);self.hits+=1
                return json.loads(raw)
            self.pending.add(key);self.misses+=1
        try:
            result=compute()
            raw=json.dumps(result,allow_nan=False,separators=(',',':')).encode()
            with self.condition:
                if len(raw)<=self.max_bytes:
                    while self.values and (self.bytes+len(raw)>self.max_bytes or len(self.values)>=self.max_entries):self._remove(next(iter(self.values)))
                    self.values[key]=(raw,time.monotonic()+self.ttl);self.bytes+=len(raw)
            return result
        finally:
            with self.condition:self.pending.remove(key);self.condition.notify_all()

    def _remove(self,key):
        raw,_=self.values.pop(key);self.bytes-=len(raw)

    def clear(self):
        with self.condition:self.values.clear();self.bytes=0

    def stats(self):
        with self.condition:return dict(entries=len(self.values),bytes=self.bytes,hits=self.hits,misses=self.misses,max_bytes=self.max_bytes,ttl_seconds=self.ttl)


cache=ResultCache()
