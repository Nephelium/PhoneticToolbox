"""Publish explicitly authorized static website files, preserving other site routes.

The SSH helper is supplied from outside the repository and uses the existing
Windows credential vault. No credential is accepted as an argument or logged.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, io, json, re, tarfile, uuid

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'temporary-site'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    files = {'site.css': (SITE / 'site.css').read_bytes(), 'site.js': (SITE / 'site.js').read_bytes()}
    html = (SITE / 'index.html').read_text('utf-8')
    names = set(re.findall(r'(?:src|href)="(assets/[^"?#]+)', html))
    names.update(re.findall(r"url\(['\"]?(assets/[^)'\"?#]+)", files['site.css'].decode()))
    names.update('assets/licenses/' + x for x in ['index.html', 'Doulos-OFL.txt', 'ZCOOL-OFL.txt'])
    names.update(['assets/audio/provenance.json', 'assets/spectrogram-examples/provenance.json', 'assets/release.json', 'assets/downloads.json'])
    names.add('assets/spectrogram-examples/comparison.json')
    names.update(str(p.relative_to(SITE)).replace('\\','/') for p in (SITE/'assets/releases').glob('*.json'))
    for name in sorted(names):
        path = SITE / name
        assert path.resolve().is_relative_to(SITE.resolve()) and path.is_file()
        assert path.suffix in {'.png', '.jpg', '.svg', '.ttf', '.html', '.txt', '.wav', '.json'}
        files[name] = path.read_bytes()
    release = 'r' + digest(html.encode() + b''.join(k.encode() + files[k] for k in sorted(files)))[:16]
    prefix = '/site-releases/' + release + '/'
    html = re.sub(r'((?:src|href)=")(assets/|site\.css|site\.js)', lambda m: m[1] + prefix + m[2], html)
    files['index.html'] = html.encode('utf-8')
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
(backup / 'nginx.conf').write_bytes(oldconf); (backup / 'index.html').write_bytes(oldindex)
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
# Preserve the deployed policy and allow only the official tutorial player frame.
policy_match = re.search(r'add_header Content-Security-Policy "([^"]*)" always;', oldconf.decode())
assert policy_match
policy = policy_match[1]
directives = [part.strip() for part in policy.split(';') if part.strip()]
frame_index = next((i for i, part in enumerate(directives) if part.split()[0] == 'frame-src'), None)
if frame_index is None:
    directives.append('frame-src https://player.bilibili.com')
else:
    values = [value for value in directives[frame_index].split()[1:] if value != "'none'"]
    if 'https://player.bilibili.com' not in values:
        values.append('https://player.bilibili.com')
    directives[frame_index] = 'frame-src ' + ' '.join(values)
policy = '; '.join(directives)
newconf, count = re.subn(r'add_header Content-Security-Policy "[^"]*" always;', 'add_header Content-Security-Policy "' + policy + '" always;', oldconf.decode())
assert count == 1
assert 'root /var/www/phonetictoolbox-coming-soon;' in newconf
try:
    atomic(conf, newconf.encode())
    subprocess.run(['nginx', '-t'], check=True, capture_output=True)
    subprocess.run(['systemctl', 'reload', 'nginx'], check=True, capture_output=True)
    atomic(root / 'index.html', (target / 'index.html').read_bytes())
    assert catalog_before == (hashlib.sha256(catalog.read_bytes()).hexdigest() if catalog.exists() else None)
    subprocess.run(['systemctl', 'is-active', '--quiet', 'ptb-feedback.service'], check=True)
except Exception:
    atomic(conf, oldconf); atomic(root / 'index.html', oldindex)
    subprocess.run(['nginx', '-t'], check=True, capture_output=True)
    subprocess.run(['systemctl', 'reload', 'nginx'], check=True, capture_output=True)
    raise
receipt = {'success': True, 'release': release, 'backup': str(backup), 'index_sha256': hashlib.sha256((root / 'index.html').read_bytes()).hexdigest(), 'files': len(PAYLOAD['files']), 'papers_catalog_unchanged': True, 'feedback_active': True, 'nginx_configuration_change': 'Preserved existing CSP; frame-src permits only the official Bilibili tutorial player in addition to existing frame sources'}
(backup / 'receipt.json').write_text(json.dumps(receipt, indent=2))
archive.unlink()
print(json.dumps(receipt))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssh-helper', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    release, manifest, archive = build()
    operation = 'website-' + uuid.uuid4().hex
    out = ROOT / 'output/validation/website' / operation
    out.mkdir(parents=True)
    (out / 'manifest.json').write_text(json.dumps({'release': release, 'files': manifest}, indent=2), 'utf-8')
    if not args.apply:
        print(json.dumps({'prepared': True, 'release': release, 'files': len(manifest), 'bytes': len(archive), 'out': str(out)})); return
    assert args.ssh_helper and args.ssh_helper.is_file()
    import sys
    sys.path.insert(0, str(args.ssh_helper.parent))
    spec = importlib.util.spec_from_file_location('website_private_ssh', args.ssh_helper)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    remote_archive = '/home/admin/website-upload-' + operation + '.tar.gz'
    payload = {'release': release, 'files': manifest, 'archive': remote_archive, 'archive_sha256': digest(archive), 'operation': operation}
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
