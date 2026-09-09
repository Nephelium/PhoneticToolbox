"""Explicit preallocation bounds for M01 I/O. Host quota integration is M01-F."""
import io
from dataclasses import dataclass
from phonetic_core.ports.errors import BackendAborted


class LimitError(BackendAborted, ValueError): pass
class FormatError(ValueError): pass
class Cancelled(BackendAborted): pass


@dataclass(frozen=True)
class Limits:
    input_bytes: int = 16_000_000
    samples: int = 2_000_000
    channels: int = 8
    text_bytes: int = 2_000_000
    text_items: int = 100_000
    tiers: int = 64
    cells: int = 200_000
    output_bytes: int = 16_000_000
    xml_bytes: int = 32_000_000
    process_bytes: int = 512_000_000
    timeout_seconds: float = 60.

    def __post_init__(self):
        for name,value in vars(self).items():
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not 0<value<float('inf'):
                raise ValueError('Invalid limit: '+name)
            if name != 'timeout_seconds' and not isinstance(value,int):
                raise ValueError('Integer limit required: '+name)


class LimitedBuffer(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        if not isinstance(limit,int) or limit<0: raise ValueError('Invalid buffer limit')
        self.limit=limit

    def write(self, data):
        if self.tell()+len(data)>self.limit: raise LimitError('output_bytes_exceeded')
        return super().write(data)

    def seek(self, offset, whence=0):
        position=offset if whence==0 else self.tell()+offset if whence==1 else len(self.getbuffer())+offset
        if not 0<=position<=self.limit: raise LimitError('output_seek_exceeded')
        return super().seek(offset,whence)

    def truncate(self, size=None):
        if size is not None and not 0<=size<=self.limit: raise LimitError('output_truncate_exceeded')
        return super().truncate(size)
