"""Validate and atomically publish an explicitly authorized public announcement feed."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys
from urllib.parse import urlsplit, unquote

LIMIT = 65536


def valid_text(value, limit):
    return (isinstance(value, str) and bool(value.strip())
            and len(value.encode('utf-16-le')) // 2 <= limit
            and not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', value))


def validate_feed(raw):
    if len(raw) > LIMIT:
        raise ValueError('Announcement feed exceeds 64 KiB')
    feed = json.loads(raw.decode('utf-8'))
    if not isinstance(feed, dict) or set(feed) != {'schema', 'items'} or feed['schema'] != 'ptb-announcements/1':
        raise ValueError('Invalid announcement schema')
    if not isinstance(feed['items'], list) or len(feed['items']) > 20:
        raise ValueError('Invalid announcement count')
    ids = set()
    for item in feed['items']:
        if not isinstance(item, dict) or not {'id', 'title', 'body', 'publishedAt'} <= set(item) or set(item) - {'id', 'title', 'body', 'publishedAt', 'expiresAt', 'minVersion', 'maxVersion', 'platforms', 'action'}:
            raise ValueError('Invalid announcement fields')
        if not isinstance(item['id'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', item['id']) or item['id'] in ids:
            raise ValueError('Invalid or duplicate announcement ID')
        ids.add(item['id'])
        for name, limit in [('title', 160), ('body', 8000)]:
            if not valid_text(item[name], limit):
                raise ValueError('Invalid announcement text')
        dates = {}
        for name in ['publishedAt', 'expiresAt']:
            if name not in item:
                continue
            value = item[name]
            if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{3})?Z', value):
                raise ValueError('Invalid UTC announcement date')
            dates[name] = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if 'expiresAt' in dates and dates['expiresAt'] <= dates['publishedAt']:
            raise ValueError('Announcement expiry must follow publication')
        versions = {}
        for name in ['minVersion', 'maxVersion']:
            if name not in item:
                continue
            value = item[name]
            if not isinstance(value, str) or not re.fullmatch(r'(?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3})', value):
                raise ValueError('Invalid announcement version')
            versions[name] = tuple(map(int, value.split('.')))
        if 'minVersion' in versions and 'maxVersion' in versions and versions['minVersion'] > versions['maxVersion']:
            raise ValueError('Invalid announcement version interval')
        if 'platforms' in item and (not isinstance(item['platforms'], list) or not item['platforms'] or any(p not in ['windows', 'macos', 'linux', 'web'] for p in item['platforms']) or len(set(item['platforms'])) != len(item['platforms'])):
            raise ValueError('Invalid announcement platforms')
        if 'action' in item:
            action = item['action']
            if not isinstance(action, dict) or set(action) != {'label', 'url'} or not valid_text(action['label'], 80) or not isinstance(action['url'], str) or len(action['url']) > 2048:
                raise ValueError('Invalid announcement action')
            url = urlsplit(action['url'])
            path = unquote(url.path, errors='strict')
            if (url.scheme != 'https' or url.netloc != 'www.phonetictoolbox.com' or url.query
                    or re.search(r'[\s\\]', action['url']) or re.search(r'[\\\x00-\x1f]', path)
                    or re.search(r'\.(?:exe|msi|msix|zip|7z|dmg|pkg|app|apk|ipa|mobileconfig|bat|cmd|ps1|sh)(?:/|$)', path, re.I)):
                raise ValueError('Only official HTTPS announcement actions are allowed')
    return feed


REMOTE = r'''
import base64,hashlib,json,os,sys
from pathlib import Path
p=json.loads(sys.stdin.readline());root=Path('/var/www/phonetictoolbox-announcements');target=root/'v1.json'
assert root.parent.resolve()==root.parent and not root.is_symlink() and not target.is_symlink()
raw=base64.b64decode(p['base64'],validate=True);assert len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==p['sha256']
actual=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else 'absent'
assert actual==p['expected'],'Announcement changed concurrently; original preserved'
if not root.exists():root.mkdir(mode=0o755)
assert root.resolve()==root
pending=root/('v1.json.'+p['sha256'][:16]+'.pending');assert not pending.exists()
with pending.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
pending.chmod(0o644)
assert hashlib.sha256(pending.read_bytes()).hexdigest()==p['sha256']
assert (hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else 'absent')==p['expected']
os.replace(pending,target)
print(json.dumps({'success':True,'path':str(target),'bytes':len(raw),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feed', type=Path, required=True)
    parser.add_argument('--ssh-helper', type=Path)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    raw = args.feed.read_bytes()
    feed = validate_feed(raw)
    digest = hashlib.sha256(raw).hexdigest()
    result = dict(applied=False, items=len(feed['items']), bytes=len(raw), sha256=digest)
    if args.apply:
        if args.ssh_helper is None or not re.fullmatch(r'(?:absent|[0-9a-f]{64})', args.expected_sha256 or ''):
            parser.error('--apply requires --ssh-helper and --expected-sha256')
        import base64
        sys.path.insert(0, str(args.ssh_helper.resolve().parent))
        spec = importlib.util.spec_from_file_location('announcement_ssh', args.ssh_helper.resolve())
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with helper.connect() as client:
            stdin, stdout, stderr = client.exec_command('sudo -n python3 -c ' + shlex.quote(REMOTE), timeout=30)
            stdin.write(json.dumps(dict(base64=base64.b64encode(raw).decode('ascii'), sha256=digest, expected=args.expected_sha256)) + '\n')
            stdin.flush(); stdin.channel.shutdown_write()
            output = stdout.read(); error = stderr.read()
            if stdout.channel.recv_exit_status():
                raise RuntimeError(error.decode('utf-8', errors='replace'))
            result.update(json.loads(output), applied=True)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
