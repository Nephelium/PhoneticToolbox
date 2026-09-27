"""Fixed child entry; bounded input and output, inherited process-group limits."""
import json
import os
import struct
import sys
from pathlib import Path

def run():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    with open(sys.argv[2],'wb',buffering=0) as stream:
        try:
            from .m07_science import compute,sha,encode
            from .managed_scratch import ReservedNativeScratch
            from .native.reaper import Reaper
            from .io.limits import Limits
            path=Path(sys.argv[1])
            if path.stat().st_size>49_000_000:raise ValueError('m07_input_budget')
            with path.open('rb') as handle:
                header=json.loads(handle.readline(65536));blobs={}
                for item in header['inputs']:
                    n=item['size_bytes']
                    if type(n)!=int or not 0<n<=32_000_000:raise ValueError('m07_input_budget')
                    raw=handle.read(n)
                    if len(raw)!=n or sha(raw)!=item['sha256']:raise ValueError('m07_input_changed')
                    blobs[item['role']]=raw
                if handle.read(1):raise ValueError('m07_invalid_bundle')
            if header.get('native_scratch'):
                with ReservedNativeScratch(header['native_scratch'],400_000) as scratch:
                    native=Reaper(header['reaper_binary'],scratch,Limits(input_bytes=400_000,output_bytes=2_000_000,process_bytes=1_000_000_000,timeout_seconds=30))
                    files=compute(header,blobs,native)
            else:files=compute(header,blobs)
            payload=b''.join(v for _,v in files)
            encoded=encode(dict(kind='prepared_m07',request_hash=header['request_hash'],files=[dict(name=n,size=len(v),sha256=sha(v)) for n,v in files]))
            if len(payload)+len(encoded)+4>64_000_000:raise ValueError('m07_output_budget')
        except Exception as exc:
            from .m07_errors import public_error
            encoded=json.dumps(dict(error=public_error(exc))).encode();payload=b''
        stream.write(struct.pack('<I',len(encoded)));stream.write(encoded)
        for offset in range(0,len(payload),65536):stream.write(payload[offset:offset+65536])

if __name__=='__main__':run()
