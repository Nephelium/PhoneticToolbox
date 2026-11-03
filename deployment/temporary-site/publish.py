"""Publish explicitly authorized static website files, preserving other site routes.

The SSH helper is supplied from outside the repository and uses the existing
Windows credential vault. No credential is accepted as an argument or logged.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, io, json, posixpath, re, sys, tarfile, uuid
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'temporary-site'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    # Generated pages must be current; publishing never falls back to old HTML.
    sys.path.insert(0, str(ROOT/'scripts'))
    from build_website_i18n import build as verify_languages, available_languages
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        verify_languages(check=True)
    baseline_spec = importlib.util.spec_from_file_location('website_audio_baseline', ROOT / 'scripts/website_audio_baseline.py')
    baseline = importlib.util.module_from_spec(baseline_spec); baseline_spec.loader.exec_module(baseline)
    baseline.verify(SITE)
    sys.path.insert(0, str(ROOT/'scripts'))
    from build_website_gl_comparison import verify as verify_three_way
    verify_three_way(SITE)
    directories = [row['directory'] for row in available_languages()[1:]]
    source_names = [
        'site.css', 'site.js', 'ime-downloads.css', 'ime-downloads.js',
        'site-languages.css', 'site-languages.js']
    source_names.extend(directory+'/'+name for directory in directories for name in ['index.html','site.js','ime-downloads.js'])
    files = {name: (SITE / name).read_bytes() for name in source_names}
    html = (SITE / 'index.html').read_text('utf-8')
    names = set(re.findall(r'(?:src|href|poster)="(assets/[^"?#]+)', html))
    if (SITE/'assets/videos/provenance.json').is_file():
        names.add('assets/videos/provenance.json')
    names.update(re.findall(r"url\(['\"]?(assets/[^)'\"?#]+)", files['site.css'].decode()))
    names.update('assets/licenses/' + x for x in ['index.html', 'Doulos-OFL.txt', 'ZCOOL-OFL.txt'])
    names.update(['assets/audio/provenance.json', 'assets/spectrogram-examples/provenance.json', 'assets/release.json', 'assets/downloads.json'])
    names.add('assets/spectrogram-examples/comparison.json')
    names.update(['assets/spectrogram-examples/basic-gl-baseline.json', 'assets/feature-icons.provenance.json'])
    names.add('assets/spectrogram-examples/three-way-baseline.json')
    if (SITE/'assets/spectrogram-examples/four-way-baseline.json').is_file():
        names.add('assets/spectrogram-examples/four-way-baseline.json')
    if (SITE/'assets/spectrogram-examples/five-way-baseline.json').is_file():
        names.add('assets/spectrogram-examples/five-way-baseline.json')
    # Retain all established listening assets, including older standalone examples.
    names.update('assets/audio/'+row['file'] for row in json.loads((SITE/'assets/audio/provenance.json').read_text('utf-8'))['files'])
    names.update(str(p.relative_to(SITE)).replace('\\','/') for p in (SITE/'assets/releases').glob('*.json'))
    for name in sorted(names):
        path = SITE / name
        assert path.resolve().is_relative_to(SITE.resolve()) and path.is_file()
        assert path.suffix in {'.png', '.jpg', '.svg', '.ttf', '.html', '.txt', '.wav', '.json', '.mp4'}
        files[name] = path.read_bytes()
    release = 'r' + digest(html.encode() + b''.join(k.encode() + files[k] for k in sorted(files)))[:16]
    prefix = '/site-releases/' + release + '/'
    def public_page(source, directory=''):
        def rewrite(match):
            attribute, value = match[1], match[2]
            parsed=urlsplit(value)
            if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith('/'):
                return match[0]
            relative=posixpath.normpath(posixpath.join(directory,parsed.path))
            suffix=('?'+parsed.query if parsed.query else '')+('#'+parsed.fragment if parsed.fragment else '')
            if relative=='index.html':
                return attribute+'/'
            for language in directories:
                if relative==language+'/index.html':
                    return attribute+'/'+language+'/'+suffix
            if relative.startswith('assets/') or relative in files:
                return attribute+prefix+relative+suffix
            return match[0]
        return re.sub(r'((?:src|href|poster)=")([^"]+)',rewrite,source)
    files['index.html'] = public_page(html).encode('utf-8')
    for directory in directories:
        name=directory+'/index.html'
        files[name] = public_page(files[name].decode('utf-8'),directory).encode('utf-8')
    manifest = {name: {'bytes': len(data), 'sha256': digest(data)} for name, data in files.items()}
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as tar:
        for name, data in files.items():
            entry = tarfile.TarInfo(name); entry.size = len(data); entry.mode = 0o644
            tar.addfile(entry, io.BytesIO(data))
    return release, manifest, archive.getvalue()


REMOTE = r'''
from pathlib import Path
import hashlib, io, json, os, re, shutil, subprocess, tarfile
root = Path('/var/www/phonetictoolbox-coming-soon')
conf = Path('/etc/nginx/conf.d/phonetictoolbox-coming-soon.conf')
assert root.resolve() == root and conf.is_file()
release = PAYLOAD['release']; assert re.fullmatch(r'r[0-9a-f]{16}', release)
archive = Path(PAYLOAD['archive']); assert archive.parent == Path('/home/admin') and archive.name.startswith('website-upload-')
data = archive.read_bytes(); assert hashlib.sha256(data).hexdigest() == PAYLOAD['archive_sha256']
target = root / 'site-releases' / release
assert not target.exists()
backup = Path('/var/backups/phonetictoolbox-website') / PAYLOAD['operation']
backup.mkdir(parents=True, mode=0o700); os.chmod(backup, 0o700)
oldconf = conf.read_bytes(); oldindex = (root / 'index.html').read_bytes()
assert hashlib.sha256(oldindex).hexdigest() == PAYLOAD['expected_state']['index_sha256'], 'Homepage changed after review'
feedback_pid = subprocess.check_output(['systemctl', 'show', 'ptb-feedback.service', '--property=MainPID', '--value'], text=True).strip()
assert feedback_pid.isdigit() and int(feedback_pid) > 0
localized = [name for name in PAYLOAD['files'] if re.fullmatch(r'[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*/index.html',name)]
oldlocalized = {}
for name in localized:
    destination = root / name
    assert destination.resolve().is_relative_to(root)
    oldlocalized[name] = destination.read_bytes() if destination.is_file() else None
    actual = hashlib.sha256(oldlocalized[name]).hexdigest() if oldlocalized[name] is not None else None
    assert actual == PAYLOAD['expected_state']['localized'][Path(name).parent.name], 'Localized page changed after review'
(backup / 'nginx.conf').write_bytes(oldconf); (backup / 'index.html').write_bytes(oldindex)
for name, content in oldlocalized.items():
    if content is not None:
        (backup / ('index.'+Path(name).parent.name+'.html')).write_bytes(content)
catalog = root / 'papers/catalog.json'
catalog_before = hashlib.sha256(catalog.read_bytes()).hexdigest() if catalog.exists() else None
target.mkdir(parents=True, mode=0o755)
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
    assert set(tar.getnames()) == set(PAYLOAD['files'])
    for member in tar.getmembers():
        assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
        content = tar.extractfile(member).read(); expected = PAYLOAD['files'][member.name]
        assert len(content) == expected['bytes'] and hashlib.sha256(content).hexdigest() == expected['sha256']
        destination = target / member.name
        assert destination.resolve().is_relative_to(target.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content); os.chmod(destination, 0o644)
def atomic(path, content):
    temp = path.with_name(path.name + '.' + PAYLOAD['operation'] + '.tmp')
    temp.write_bytes(content); os.chmod(temp, 0o644); os.replace(temp, path)
# This publication only changes static pages. Existing routing and CSP are kept.
assert (root / 'index.html').read_bytes() == oldindex, 'Homepage changed during staging'
assert conf.read_bytes() == oldconf, 'Server configuration changed during staging'
for name, content in oldlocalized.items():
    destination = root / name
    assert (destination.read_bytes() if destination.is_file() else None) == content
try:
    for name in localized:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic(destination, (target / name).read_bytes())
    atomic(root / 'index.html', (target / 'index.html').read_bytes())
    assert catalog_before == (hashlib.sha256(catalog.read_bytes()).hexdigest() if catalog.exists() else None)
    assert conf.read_bytes() == oldconf
    assert subprocess.check_output(['systemctl', 'show', 'ptb-feedback.service', '--property=MainPID', '--value'], text=True).strip() == feedback_pid
    subprocess.run(['systemctl', 'is-active', '--quiet', 'ptb-feedback.service'], check=True)
except Exception:
    atomic(root / 'index.html', oldindex)
    for name, content in oldlocalized.items():
        destination = root / name
        if content is not None:
            atomic(destination, content)
        elif destination.is_file():
            destination.unlink()
    raise
receipt = {'success': True, 'release': release, 'backup': str(backup), 'index_sha256': hashlib.sha256((root / 'index.html').read_bytes()).hexdigest(), 'files': len(PAYLOAD['files']), 'papers_catalog_unchanged': True, 'feedback_active': True, 'feedback_pid_unchanged': True, 'nginx_configuration_changed': False, 'default_language': 'zh-CN', 'localized_pages': localized}
(backup / 'receipt.json').write_text(json.dumps(receipt, indent=2))
archive.unlink()
print(json.dumps(receipt))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssh-helper', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--expected-state', type=Path, help='Reviewed server snapshot containing index_sha256 and localized page hashes')
    args = parser.parse_args()
    release, manifest, archive = build()
    operation = 'website-' + uuid.uuid4().hex
    out = ROOT / 'output/validation/website' / operation
    out.mkdir(parents=True)
    (out / 'manifest.json').write_text(json.dumps({'release': release, 'files': manifest}, indent=2), 'utf-8')
    if not args.apply:
        print(json.dumps({'prepared': True, 'release': release, 'files': len(manifest), 'bytes': len(archive), 'out': str(out)})); return
    assert args.ssh_helper and args.ssh_helper.is_file()
    assert args.expected_state and args.expected_state.is_file(), '--apply requires the reviewed server snapshot'
    expected = json.loads(args.expected_state.read_text('utf-8'))
    assert re.fullmatch(r'[0-9a-f]{64}', expected['index_sha256'])
    expected = {'index_sha256': expected['index_sha256'], 'localized': expected['localized']}
    import sys
    sys.path.insert(0, str(args.ssh_helper.parent))
    spec = importlib.util.spec_from_file_location('website_private_ssh', args.ssh_helper)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    remote_archive = '/home/admin/website-upload-' + operation + '.tar.gz'
    payload = {'release': release, 'files': manifest, 'archive': remote_archive, 'archive_sha256': digest(archive), 'operation': operation, 'expected_state': expected}
    with helper.connect() as client:
        with client.open_sftp() as sftp:
            with sftp.file(remote_archive, 'wb') as handle: handle.write(archive)
            sftp.chmod(remote_archive, 0o600)
        stdin, stdout, stderr = client.exec_command('sudo -n python3 -', timeout=90)
        stdin.write('PAYLOAD = ' + repr(payload) + '\n' + REMOTE); stdin.flush(); stdin.channel.shutdown_write()
        result = stdout.read().decode('utf-8'); error = stderr.read().decode('utf-8')
        code = stdout.channel.recv_exit_status()
        if code: raise RuntimeError('Publication failed; remote rollback attempted: ' + error)
        receipt = json.loads(result)
    (out / 'deployment.json').write_text(json.dumps(receipt, indent=2), 'utf-8')
    print(json.dumps({'out': str(out), **receipt}))


if __name__ == '__main__': main()
