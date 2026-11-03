"""Exercise the actual final EXE's language controls and complete offline manuals."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import zipfile

from portable_cache_qa import seed_portable_profile
from verification_artifacts import verification_run, write_json
from pypdf import PdfWriter
from owned_rejecting_proxy import RejectingProxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--english-pack', type=Path, required=True)
    parser.add_argument('--japanese-pack', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    exe = args.exe.resolve(strict=True)
    with exe.open('rb') as stream:
        identity = {'path': str(exe), 'bytes': exe.stat().st_size,
                    'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    packs = {}
    for locale, file in (('en', args.english_pack), ('ja', args.japanese_pack)):
        file = file.resolve(strict=True)
        with zipfile.ZipFile(file) as archive:
            manifest = json.loads(archive.read('language-pack.json'))
            project = json.loads(archive.read('project.json'))
            titles, chapter_media = {}, {}
            for chapter in project['chapters']:
                content = json.loads(archive.read(chapter['path']))
                titles[chapter['id']] = content['title']
                media = {'image': [], 'audio': [], 'video': []}
                def visit(node):
                    if isinstance(node, dict):
                        if node.get('type') in media:
                            media[node['type']].append(node['attrs']['assetId'])
                        for child in node.values():
                            visit(child)
                    elif isinstance(node, list):
                        for child in node:
                            visit(child)
                visit(content['body'])
                chapter_media[chapter['id']] = media
        assert manifest['locale'] == locale and len(titles) == 22
        with file.open('rb') as stream:
            packs[locale] = {'path': str(file), 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest(),
                             'sourceDigest': manifest['sourceDigest'], 'chapterTitles': titles,
                             'chapterMedia': chapter_media}
    with verification_run('app-i18n-package', 'final-exe',
            'Actual final EXE, visible Windows/default GPU, private data; 19 modules x 3 languages, existing validation, real full 22-chapter ZIP imports and 21 desktop chapters offline reopen; per-process HTTP(S) rejecting proxy') as (out, scratch), RejectingProxy(out / 'network-denials.json') as proxy:
        profile = scratch / 'profile'
        profile.mkdir()
        seed_portable_profile(profile)
        library = scratch / 'library'
        library.mkdir()
        writer = PdfWriter()
        for _ in range(4):
            writer.add_blank_page(width=612, height=792)
        with (library / 'Owned fullscreen fixture.pdf').open('wb') as stream:
            writer.write(stream)
        data = profile / 'PhoneticToolbox/v3'
        for folder in ('papers', 'updates'):
            (data / folder).mkdir(parents=True, exist_ok=True)
        write_json(data / 'papers/preferences.json', {'autoReceive': False, 'checkIntervalHours': 6, 'libraryRoot': str(library)})
        write_json(data / 'updates/state.json', {'autoCheck': False})
        write_json(out / 'input.json', {'exe': identity, 'packs': packs})
        for phase in ('initial', 'reopen'):
            temp = scratch / phase
            temp.mkdir()
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(('PTB_', 'PYTHON', 'CONDA', '_PYI', 'QT_')) and k != 'VIRTUAL_ENV'
                   and k.upper() not in {'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY'}}
            env.update(LOCALAPPDATA=str(profile), TEMP=str(temp), TMP=str(temp),
                       PATH=os.pathsep.join((os.environ['SystemRoot'] + '/System32', os.environ['SystemRoot'])),
                       QT_QPA_PLATFORM='windows', QTWEBENGINE_REMOTE_DEBUGGING=str(port),
                       QTWEBENGINE_CHROMIUM_FLAGS='--mute-audio --remote-allow-origins=* --disable-quic --proxy-server=' + proxy.url,
                       PTB_OWNED_BOOTSTRAP_LOG=str(out / (phase + '-bootstrap.log')))
            env.update(proxy.environment())
            with (out / (phase + '-process.log')).open('wb') as log:
                child = subprocess.Popen([str(exe)], cwd=temp, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    with (out / (phase + '-runner.log')).open('wb') as runner_log:
                        result = subprocess.run(['node', str(root / 'scripts/verify_app_i18n_package.mjs'),
                                                 str(port), str(out), phase], cwd=root,
                                                stdout=runner_log, stderr=subprocess.STDOUT, timeout=900)
                    assert result.returncode == 0, (out / (phase + '-runner.log')).read_text('utf-8')[-6000:]
                    assert child.wait(timeout=45) == 0
                finally:
                    if child.poll() is None:
                        subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'],
                                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                        child.wait(timeout=15)
            assert json.loads((data / 'language.json').read_text('utf-8'))['locale'] == 'ja'
        assert proxy.requests and any(row['target'].startswith('www.phonetictoolbox.com:') for row in proxy.requests)
        assert all(row['forwarded'] is False for row in proxy.requests)
        write_json(out / 'report.json', {'success': True, 'identity': identity,
                   'phases': ['initial', 'reopen'], 'network': 'Per-process urllib/Chromium HTTP(S) rejection; local API loopback retained; no OS/firewall changes',
                   'scope': 'Actual final Windows EXE on this computer; visible default GPU; no physical audio device or separate clean OS'})
        print(json.dumps({'success': True, 'report': str(out / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
