"""Reproduce an audited stored ZIP from its already-published identical EXE.

This changes transport only: archive metadata, size and SHA must match locally.
"""
import base64
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import shutil
import tempfile
import zipfile

NAMES = ('PhoneticToolbox/PhoneticToolbox.exe', 'PhoneticToolbox/application.json')


def write_archive(destination, executable, recipe):
    import base64
    import shutil
    import zipfile
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_STORED) as archive:
        archive.comment = base64.b64decode(recipe['comment'])
        for item in recipe['members']:
            info = zipfile.ZipInfo(item['filename'], tuple(item['date_time']))
            for name, value in item['attributes'].items():
                setattr(info, name, value)
            info.extra = base64.b64decode(item['extra'])
            info.comment = base64.b64decode(item['comment'])
            with archive.open(info, 'w') as target:
                if item['filename'] == 'PhoneticToolbox/PhoneticToolbox.exe':
                    with open(executable, 'rb') as source:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                else:
                    target.write(base64.b64decode(recipe['application']))


def prepare_recipe(local_zip, local_exe, owned_work):
    local_zip, local_exe = Path(local_zip), Path(local_exe)
    with zipfile.ZipFile(local_zip) as archive:
        infos = archive.infolist()
        if tuple(info.filename for info in infos) != NAMES or any(info.compress_type != zipfile.ZIP_STORED for info in infos):
            raise ValueError('Only the audited two-member stored update layout can be reproduced')
        application = archive.read(NAMES[1])
        if len(application) > 65536:
            raise ValueError('Application identity is too large')
        descriptor = json.loads(application)
        with local_exe.open('rb') as stream:
            expected = dict(size=local_exe.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
        if descriptor['executable'] != expected:
            raise ValueError('ZIP application descriptor differs from the local EXE')
        with archive.open(NAMES[0]) as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected['sha256']:
                raise ValueError('ZIP EXE bytes differ')
        attributes = ('compress_type', 'create_system', 'create_version', 'extract_version',
                      'reserved', 'flag_bits', 'volume', 'internal_attr', 'external_attr', 'file_size')
        recipe = {'comment': base64.b64encode(archive.comment).decode(),
                  'application': base64.b64encode(application).decode(), 'exe': expected,
                  'members': [{'filename': info.filename, 'date_time': info.date_time,
                               'attributes': {name: getattr(info, name) for name in attributes},
                               'extra': base64.b64encode(info.extra).decode(),
                               'comment': base64.b64encode(info.comment).decode()} for info in infos]}
    # Test the actual 500 MB input recipe before asking the server to reproduce it.
    with tempfile.TemporaryDirectory(prefix='zip-recipe-', dir=owned_work) as scratch:
        derived = Path(scratch) / 'check.zip'
        write_archive(derived, local_exe, recipe)
        with local_zip.open('rb') as original, derived.open('rb') as generated:
            if hashlib.file_digest(original, 'sha256').digest() != hashlib.file_digest(generated, 'sha256').digest():
                raise ValueError('Stored ZIP recipe does not reproduce exact local bytes')
    return recipe


def publish_derived(client, row, source_relative, recipe):
    remote = inspect.getsource(write_archive) + r'''
import hashlib,json,os,sys
from pathlib import Path
p=json.loads(sys.stdin.readline())
root=Path('/var/www/phonetictoolbox-coming-soon')
target=root/p['row']['relative'];source=root/p['source']
assert target.resolve().is_relative_to(root/'releases') and source.resolve().is_relative_to(root/'releases')
assert not target.is_symlink() and not source.is_symlink() and target.parent==source.parent
assert source.name=='PhoneticToolbox-windows-x64.exe' and target.suffix=='.zip'
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert source.stat().st_size==p['recipe']['exe']['size'] and digest(source)==p['recipe']['exe']['sha256']
if target.exists():
    assert target.stat().st_size==p['row']['size'] and digest(target)==p['row']['sha256']
else:
    temporary=target.with_name(target.name+'.'+p['row']['sha256'][:16]+'.derived.tmp')
    assert not temporary.exists() and not temporary.is_symlink()
    try:
        write_archive(temporary,source,p['recipe'])
        assert temporary.stat().st_size==p['row']['size'] and digest(temporary)==p['row']['sha256']
        os.chmod(temporary,0o644);os.replace(temporary,target)
    finally:
        if temporary.is_file():temporary.unlink()
print(json.dumps({'success':True,'path':str(target),'sha256':digest(target),'size':target.stat().st_size,'derivedFromPublishedExe':True}))
'''
    stdin, stdout, stderr = client.exec_command('sudo -n python3 -c ' + shlex.quote(remote), timeout=600)
    stdin.write(json.dumps({'row': row, 'source': source_relative, 'recipe': recipe}) + '\n')
    stdin.flush(); stdin.channel.shutdown_write()
    raw, error = stdout.read().decode(), stderr.read().decode()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error or raw)
    result = json.loads(raw)
    if not result['success'] or result['sha256'] != row['sha256'] or result['size'] != row['size']:
        raise ValueError('Derived server ZIP identity mismatch')
    return result
