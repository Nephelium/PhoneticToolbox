"""Fixed M14 child, in-memory output framing compatible with the bounded pipe."""
import hashlib
import json
import os
import struct
import sys
from pathlib import Path


def run():
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    with open(sys.argv[2],'wb',buffering=0) as stream:
        try:
            from .m14_jobs import execute
            raw=Path(sys.argv[1]).read_bytes();header,payload=raw.split(b'\n',1);header=json.loads(header)
            if hashlib.sha256(payload).hexdigest()!=header['sha256']:raise ValueError('m14_input_changed')
            outputs=execute(payload,header['name'],header['config'])
            manifest=dict(kind='prepared_m14',input_sha256=header['sha256'],files=[dict(name=n,size=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in outputs.items()])
            encoded=json.dumps(manifest,ensure_ascii=False).encode();stream.write(struct.pack('<I',len(encoded)));stream.write(encoded)
            for value in outputs.values():stream.write(value)
        except Exception as e:
            code=str(e);code=code if code.startswith('m14_') and len(code)<80 else 'm14_execution_failed'
            encoded=json.dumps(dict(error=code)).encode();stream.write(struct.pack('<I',len(encoded)));stream.write(encoded)


if __name__=='__main__':run()
