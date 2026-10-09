"""Wrap one frozen application with a small persistent-cache launcher.

Opaque XZ segments precede the launcher's CArchive, so PyInstaller only expands
its small bootstrap. Large segments are read on demand by absolute offsets.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from PyInstaller.archive.readers import CArchiveReader


def identity(path):
    with path.open('rb') as file: sha = hashlib.file_digest(file, 'sha256').hexdigest()
    return {'size': path.stat().st_size, 'sha256': sha}


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def build(application, destination, work, snapshot, host_archive, *, env):
    application, destination, work, snapshot, host_archive = map(Path, (application,destination,work,snapshot,host_archive))
    internal = application / '_internal'
    science_descriptor = json.loads((internal/'desktop-bundle.json').read_text('utf8'))['runtimeArchive']
    host_descriptor = json.loads((internal/'host-archive.json').read_text('utf8'))
    science = json.loads((internal/'runtime-files.json').read_text('utf8'))['files']
    host_value = json.loads((internal/'host-files.json').read_text('utf8')); host = host_value['files']
    (internal/'runtimes').mkdir(exist_ok=True)
    (internal/'runtimes/.ptb-runtime-complete.json').write_text(json.dumps({'schema':'ptb-runtime-ready/1','archiveSha256':science_descriptor['sha256']}),'utf8')
    (internal/'.ptb-host-ready.json').write_text(json.dumps({'schema':'ptb-host-ready/2','sha256':host_descriptor['sha256'],
        'manifestSha256':host_descriptor['manifestSha256'],'restoration':{'persistent':True}}),'utf8')
    omit = {'_internal/runtime-payload.tar.xz','_internal/host-payload.tar.xz'}
    files = {p.relative_to(application).as_posix():identity(p) for p in sorted(application.rglob('*')) if p.is_file() and p.relative_to(application).as_posix() not in omit}
    combined = dict(files)
    for group in (science,host):
        for name,row in group.items():
            target = '_internal/' + name
            if target in combined: raise ValueError('Cached runtime collides with the application')
            combined[target]=row
    pack = work/'cached-launcher'; pack.mkdir()
    app_archive = pack/'application.tar.xz'
    with tarfile.open(app_archive,'w:xz',preset=9) as archive:
        for name in files:
            info=archive.gettarinfo(str(application/name),arcname=name)
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
            with (application/name).open('rb') as source: archive.addfile(info,source)
    paths={'science':internal/'runtime-payload.tar.xz','host':internal/'host-payload.tar.xz','apps':app_archive}
    config={'schema':'ptb-cache-payload/1','blobs':{name:identity(path) for name,path in paths.items()},
            'components':{name:{'id':json_hash(records),'files':records} for name,records in [('science',science),('host',host),('apps',combined)]},
            'sharedFiles':host_value['sharedFiles'],'applicationFiles':files,
            'scienceDescriptor':science_descriptor,'hostDescriptor':host_descriptor}
    descriptor=pack/'cache-payload.json'; descriptor.write_text(json.dumps(config,separators=(',',':')),'utf8')
    with (pack/'bootstrap-build.log').open('wb') as log:
        subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--onefile','--windowed',
            '--name','PhoneticToolboxBootstrap','--distpath',str(pack/'dist'),'--workpath',str(pack/'build'),
            '--specpath',str(pack),'--paths',str(snapshot/'desktop/src'),
            '--icon',str(snapshot/'PhoneticToolbox-v3.ico'),'--add-data',str(descriptor)+';.',
            '--exclude-module','PyQt6','--exclude-module','numpy',str(snapshot/'scripts/cache_launcher_entry.py')],
            cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    bootstrap=pack/'dist/PhoneticToolboxBootstrap.exe'
    reader=CArchiveReader(str(bootstrap)); offset=reader._start_offset
    with bootstrap.open('rb') as source,destination.open('xb') as out:
        out.write(source.read(offset))
        for path in paths.values():
            with path.open('rb') as raw: shutil.copyfileobj(raw,out,1024*1024)
        shutil.copyfileobj(source,out,1024*1024)
    from PyInstaller.utils.win32.winutils import update_exe_pe_checksum
    update_exe_pe_checksum(str(destination))
    if json.loads(CArchiveReader(str(destination)).extract('cache-payload.json')) != config:
        raise ValueError('Cached launcher metadata did not survive wrapping')
    report={'schema':'ptb-cached-build/1','executable':identity(destination),
            'components':{name:{'id':row['id'],'files':len(row['files']),'expandedBytes':sum(r['size'] for r in row['files'].values())} for name,row in config['components'].items()},
            'bootstrapBytes':bootstrap.stat().st_size,'blobs':config['blobs']}
    (work/'cached-launcher-report.json').write_text(json.dumps(report,indent=2),'utf8')
    return report
