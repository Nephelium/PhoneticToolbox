"""Build an isolated full-text author preview for the existing manual renderer.

The final language-pack builder remains strict. This helper only writes below
an active verification_run scratch owned by its caller and cannot be used as a
release builder. Untranslated screenshots remain shared source media until
their actual locale-controlled captures are registered.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

from .build import build
from .core import ManualError, dump, read_json, safe_file
from .localization import PACK_SCHEMA, localized_reader

ROOT = Path(__file__).resolve().parents[2]


def prepare(scratch: Path, locale: str, frontend: Path) -> dict:
    scratch = scratch.resolve()
    allowed = ROOT / 'output/validation'
    marker_file = scratch.parent / 'run.json'
    if not scratch.is_relative_to(allowed) or scratch.name != 'scratch' or not marker_file.is_file():
        raise ManualError('作者预览只允许写入当前验收的自有 scratch')
    marker = read_json(marker_file)
    if marker.get('schema') != 'ptb-test-run/1' or marker.get('owner_pid') != os.getpid() or marker.get('state') != 'active':
        raise ManualError('作者预览需要当前进程拥有的活动验收目录')
    for ancestor in (scratch, *scratch.parents):
        if ancestor.is_symlink() or getattr(ancestor, 'is_junction', lambda: False)():
            raise ManualError('作者预览不允许联接或符号链接')
    project, chapters, digest = localized_reader(ROOT / 'manual', locale)
    preview_frontend = scratch / 'frontend'
    preview_frontend.mkdir()
    shutil.copyfile(frontend / 'index.html', preview_frontend / 'index.html')
    shutil.copytree(frontend / 'assets', preview_frontend / 'assets')
    chinese = build(ROOT / 'manual', preview_frontend / 'manual', referenced_assets_only=True)
    assert read_json(preview_frontend / 'manual/project.json')['sourceDigest'] == digest
    payload = {'project.json': dump(project)}
    payload.update({descriptor['path']: dump(chapter) for descriptor, chapter in zip(project['chapters'], chapters)})
    overrides = {}
    registered = read_json(ROOT / f'manual/locales/{locale}/assets.json').get('overrides', {})
    source_assets = {asset['id']: asset for asset in read_json(ROOT / 'manual/project.json')['assets']}
    for identity, record in registered.items():
        source = source_assets[identity]
        image = safe_file(ROOT / 'manual', record['path']).read_bytes()
        if not record.get('reviewed') or record.get('locale') != locale or hashlib.sha256(image).hexdigest() != record['sha256']:
            raise ManualError('作者预览已登记截图校验失败: ' + identity)
        if hashlib.sha256(safe_file(ROOT / 'manual', source['path']).read_bytes()).hexdigest() != record['sourceSha256']:
            raise ManualError('作者预览截图中文源已经变化: ' + identity)
        name = 'assets/' + identity + Path(record['path']).suffix
        payload[name] = image
        overrides[identity] = name
    version = 'author-preview-' + digest[:12]
    manifest = {'schemaVersion': PACK_SCHEMA, 'locale': locale, 'version': version,
                'projectId': project['id'], 'sourceDigest': digest,
                'manualVersion': project['version'], 'softwareVersion': project.get('softwareVersion'),
                'distribution': 'software', 'sharedAssets': [asset['id'] for asset in project['assets']],
                'assetOverrides': overrides, 'authoringPreview': True,
                'hashes': {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}}
    payload['language-pack.json'] = dump(manifest)
    archive = scratch / ('author-preview-' + locale + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as pack:
        for name, data in payload.items():
            pack.writestr(name, data)
    raw = archive.read_bytes()
    spec = {'locale': locale, 'version': version, 'sourceDigest': digest, 'sha256': hashlib.sha256(raw).hexdigest()}
    from ptb_desktop.manual_packs import ManualPackManager
    manager = ManualPackManager(scratch / 'installed-author-preview')
    request = manager.invoke('importBegin', {'spec': spec, 'bytes': len(raw)})
    try:
        for offset in range(0, len(raw), 1024 * 1024):
            manager.invoke('importChunk', {'id': request['id'], 'offset': offset,
                                          'base64': base64.b64encode(raw[offset:offset + 1024 * 1024]).decode('ascii')})
        installed = manager.invoke('importFinish', request)
    except BaseException:
        try:
            manager.invoke('importCancel', request)
        except Exception:
            pass
        raise
    return {'frontend': preview_frontend, 'manager': manager,
            'receipt': {'authoringPreview': True, 'releaseCandidate': False, 'sourceDigest': digest,
                        'locale': locale, 'version': version, 'archiveBytes': len(raw), 'archiveSha256': spec['sha256'],
                        'chapters': len(chapters), 'registeredOverrides': len(overrides),
                        'chineseReferencedAssets': len(chinese.get('assets', [])) if isinstance(chinese.get('assets'), list) else chinese.get('copiedAssets'),
                        'isolatedInstalled': installed}}
