"""Incremental verifier/writer; publication remains the common fenced commit."""
import hashlib
import json
import struct
from .io.limits import FormatError, LimitError


class Receiver:
    def __init__(self, files, identity, sha256, *, write_chunk_bytes=65536):
        if type(write_chunk_bytes) is not int or not 0 < write_chunk_bytes <= 1_048_576:
            raise ValueError("m08_write_chunk_budget")
        self.write_chunk_bytes = write_chunk_bytes
        self.files, self.identity, self.sha256 = files, identity, sha256
        self.buffer = bytearray(); self.header_size = None; self.current = None
        self.offset = 0; self.total = 0; self.count = 0; self.names = set(); self.complete = False

    def write(self, chunk):
        self.total += len(chunk)
        if self.total > 64_000_000: raise LimitError('m08_output_budget')
        self.buffer.extend(chunk)
        while self.buffer:
            if self.complete: raise FormatError('m08_trailing_output')
            if self.current:
                header, asset, digest = self.current
                remaining = header['size']-self.offset
                # Offline storage fsyncs each authorized write. Aggregate bounded
                # pipe fragments instead of committing every tiny ReadFile result.
                if self.write_chunk_bytes > 65536 and len(self.buffer) < min(remaining, self.write_chunk_bytes):
                    return
                size = min(len(self.buffer), remaining, self.write_chunk_bytes)
                raw = bytes(self.buffer[:size]); del self.buffer[:size]
                self.files.write(self.identity, asset['id'], self.offset, raw)
                digest.update(raw); self.offset += size
                if self.offset == header['size']:
                    if digest.hexdigest() != header['sha256']: raise FormatError('m08_output_hash')
                    self.files.seal(self.identity, asset['id']); self.count += 1; self.current = None
                continue
            if self.header_size is None:
                if len(self.buffer)<4: return
                self.header_size = struct.unpack('<I',self.buffer[:4])[0]; del self.buffer[:4]
                if not 0<self.header_size<=2_000_000: raise FormatError('m08_output_header')
            if len(self.buffer)<self.header_size: return
            header = json.loads(bytes(self.buffer[:self.header_size])); del self.buffer[:self.header_size]; self.header_size=None
            if not isinstance(header,dict):raise FormatError('m08_output_header')
            if header.get('kind')=='error': raise FormatError(header.get('code','m08_execution_failed'))
            if header.get('kind')=='complete':
                if header.get('audio_sha256')!=self.sha256 or header.get('count')!=self.count or 'm08.ptb.json' not in self.names:
                    raise FormatError('m08_incomplete_output')
                self.complete=True; continue
            name=header.get('name'); size=header.get('size')
            if (header.get('kind')!='file' or not isinstance(name,str) or not name.endswith(('.wav','.ptb.json'))
                    or any(ord(c)<32 or c in '/\\:<>"|?*' for c in name) or len(name)>220
                    or name.casefold() in self.names or self.count>=257 or type(size)!=int or not 0<size<=64_000_000):
                raise FormatError('m08_invalid_output')
            self.names.add(name.casefold())
            asset=self.files.output(self.identity,name,'result',size)
            self.current=(header,asset,hashlib.sha256()); self.offset=0

    def finish(self):
        if not self.complete or self.buffer or self.current or self.header_size is not None:
            raise FormatError('m08_incomplete_output')
