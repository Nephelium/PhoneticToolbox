"""Render REVIEW COPIES only. Does not install, reload, provision TLS or deploy."""
import argparse
import json
from pathlib import Path
import re
from host import validate_config


def render(source,output,domain,release):
    if (not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]{0,250}[a-z0-9])?',domain) or
        '.' not in domain or '..' in domain or
        not re.fullmatch(r'[a-z0-9][a-z0-9.-]{0,63}',release) or '..' in release):
        raise ValueError('Invalid domain or release identifier')
    rendered={}
    for path in sorted(Path(source).glob('*.in')):
        raw=path.read_text('utf-8').replace('REPLACE_DOMAIN',domain).replace('REPLACE_RELEASE',release)
        if 'REPLACE_' in raw: raise ValueError('Unresolved target')
        rendered[path.name.removesuffix('.in')]=raw
    validate_config(json.loads(rendered['config.json']))
    Path(output).mkdir(parents=True,exist_ok=False)
    for name,raw in rendered.items():
        (Path(output)/name).write_text(raw,encoding='utf-8',newline='\n')
    return sorted(rendered)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--templates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--domain',required=True);p.add_argument('--release',required=True)
    a=p.parse_args()
    files=render(a.templates,a.output,a.domain,a.release)
    print(json.dumps({'files':files,'deployed':False}))


if __name__=='__main__':main()
