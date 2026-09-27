"""Public log view: preserve private raw diagnostics, redact host paths/secrets."""
import re
from pathlib import Path
from .components import no_links


def redact(text,roots=()):
    for index,root in enumerate(roots):
        for spelling in (str(root),str(root).replace('\\','/')):
            text=text.replace(spelling,'<task>' if index==0 else '<runtime>')
    text=re.sub(r'[A-Za-z]:[\\/][^\r\n"\']*','<local-path>',text)
    text=re.sub(r'(?i)((?:password|token|secret|api_key)\s*[:=]\s*)[^\s,;]+',r'\1<redacted>',text)
    text=re.sub(r'/home/[^/\s"\']+','/home/<user>',text)
    return text


def read_log(root,runtime=None):
    path=Path(root)/'native.log';no_links(path)
    if not path.is_file():return dict(text='',truncated=False)
    size=path.stat().st_size
    with path.open('rb') as stream:
        stream.seek(max(0,size-262144));raw=stream.read(262144)
    return dict(text=redact(raw.decode('utf8',errors='replace'),[root]+([runtime] if runtime else [])),truncated=size>262144)
