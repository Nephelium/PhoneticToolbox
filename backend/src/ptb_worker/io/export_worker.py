"""Private owned subprocess entry. Receives only host-created bounded JSON."""
import json
from pathlib import Path
import re
import struct
import sys
from .limits import Limits
from .scratch import no_links
from .parameter_exports import _build_pair


def main():
    if len(sys.argv)!=3:raise ValueError('Invalid export worker arguments')
    path=Path(sys.argv[1]);pipe=sys.argv[2]
    if path.parent!=Path.cwd() or not re.fullmatch(r'[0-9a-f]{32}\.json',path.name):raise ValueError('Invalid owned request')
    if not re.fullmatch(r'\\\\\.\\pipe\\ptb-m01-[0-9a-f]{32}',pipe):raise ValueError('Invalid owned pipe')
    no_links(path)
    if path.stat().st_size>16_000_000:raise ValueError('Request exceeds worker envelope')
    request=json.loads(path.read_bytes());limits=Limits(**request['limits'])
    pair=_build_pair(request['table'],limits)
    with open(pipe,'wb',buffering=0) as output:
        for blob in (struct.pack('<QQ',len(pair.xlsx),len(pair.sqlite)),pair.xlsx,pair.sqlite):
            for i in range(0,len(blob),4096):
                chunk=memoryview(blob)[i:i+4096]
                while chunk:
                    n=output.write(chunk)
                    if not n:raise OSError('Closed export pipe')
                    chunk=chunk[n:]


if __name__=='__main__':main()
