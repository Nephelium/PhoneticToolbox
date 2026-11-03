"""Own an isolated browser profile and synthetic packs, retaining only evidence."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import copy
import io
import math
import struct
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from verification_artifacts import verification_run
from scripts.manual.build import build
from scripts.manual.core import dump,read_json
from scripts.manual.pack_media import package_asset
from scripts.manual.localization import CATALOG_SCHEMA, build_pack, groups


def main() -> int:
    with verification_run('manual-language-packs', 'browser', 'Synthetic two-chapter packs: real HTTP download, IndexedDB installation, offline reload, bounded chapter reads, lazy image URLs; not final manual content') as (out, scratch):
        author = scratch / 'author'
        author.mkdir()
        pixel = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==')
        image = author / 'assets/public/screen.png'
        image.parent.mkdir(parents=True)
        from PIL import Image
        buffer=io.BytesIO();Image.new('RGB',(320,160),'#b8c9d7').save(buffer,format='PNG');pixel=buffer.getvalue()
        image.write_bytes(pixel)
        audio=author/'assets/public/tone.wav'
        with wave.open(str(audio),'wb')as stream:
            stream.setparams((1,2,16000,16000,'NONE','not compressed'));stream.writeframes(b''.join(struct.pack('<h',round(9000*math.sin(2*math.pi*150*i/16000)))for i in range(16000)))
        project = {'schemaVersion': 'ptb-manual/1', 'id': 'manual-language-fixture', 'title': '隔离验收说明书', 'language': 'zh-CN', 'version': 'fixture.1', 'chapters': [],
                   'assets': [{'id': 'fixture-screen', 'kind': 'image', 'path': 'assets/public/screen.png', 'distribution': 'public', 'git': True, 'sourceType': '实际应用截图', 'caption': '示例图片', 'width': 320, 'height': 160, 'sha256':hashlib.sha256(pixel).hexdigest()}]}
        project['assets'] += [{**project['assets'][0],'id':'fixture-screen-copy'},
          {**project['assets'][0],'id':'fixture-shared-image','sourceType':'','path':'assets/public/screen.png'},
          {'id':'fixture-shared-audio','kind':'audio','path':'assets/public/tone.wav','distribution':'public','git':True,'caption':'示例图片','sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),'mime':'audio/wav','sampleRate':16000,'channels':1,'duration':1}]
        (author/'locales').mkdir();(author/'locales/scientific-figures.json').write_bytes(dump({'schemaVersion':'ptb-manual-scientific-review/1','figures':{'fixture-shared-image':{'sourceSha256':hashlib.sha256(pixel).hexdigest(),'reviewed':True,'decision':'reuse-source'}}}))
        chapters = []
        for identity, title in [('getting-started', '第一章'), ('second', '第二章')]:
            chapter = {'schemaVersion': 'ptb-manual-chapter/1', 'id': identity, 'title': title, 'body': {'type': 'doc', 'content': [
                {'type': 'heading', 'attrs': {'id': 'input', 'level': 2}, 'content': [{'type': 'text', 'text': '输入设置'}]},
                {'type': 'paragraph', 'content': [{'type': 'text', 'text': '基频为 100 Hz。'}]},
                {'type': 'image', 'attrs': {'assetId': 'fixture-screen', 'caption': '示例图片'}},
            ]}}
            chapter['body']['content']=[node for node in chapter['body']['content']if identity=='getting-started'or node['type']!='image']
            if identity=='getting-started':chapter['body']['content'].append({'type':'image','attrs':{'assetId':'fixture-screen-copy','caption':'示例图片'}})
            chapter['body']['content'] += [{'type':'image','attrs':{'assetId':'fixture-shared-image','caption':'示例图片'}},{'type':'audio','attrs':{'assetId':'fixture-shared-audio','caption':'示例图片'}}]
            chapters.append(chapter)
            descriptor = {'id': identity, 'title': title, 'path': 'chapters/' + identity + '.json', 'status': 'verified'}
            project['chapters'].append(descriptor)
            target = author / descriptor['path']; target.parent.mkdir(exist_ok=True); target.write_bytes(dump(chapter))
        (author / 'project.json').write_bytes(dump(project))
        dictionaries = {
            'en': {'隔离验收说明书': 'Isolated test manual', '实际应用截图': 'Application screenshot', '示例图片': 'Example image', '第一章': 'First chapter', '第二章': 'Second chapter', '输入设置': 'Input settings', '基频为 100 Hz。': 'Fundamental frequency is 100 Hz.'},
            'ja': {'隔离验收说明书': '隔離検証用マニュアル', '实际应用截图': 'アプリの画面', '示例图片': 'サンプル画像', '第一章': '最初の章', '第二章': '次の章', '输入设置': '入力設定', '基频为 100 Hz。': '基本周波数は100 Hzです。'},
        }
        site = scratch / 'site'; site.mkdir()
        artifacts = []
        for locale, dictionary in dictionaries.items():
            folder = author / 'locales' / locale; folder.mkdir(parents=True)
            for name, units in groups(project, chapters).items():
                catalog = {'schemaVersion': CATALOG_SCHEMA, 'locale': locale, 'entries': {u.key: {'source': u.source, 'translation': dictionary[u.source]} for u in units}}
                (folder / (name + '.json')).write_bytes(dump(catalog))
            (folder / 'assets.json').write_bytes(dump({'schemaVersion': 'ptb-manual-localized-assets/1', 'locale': locale, 'overrides': {identity: {'path': 'assets/public/screen.png', 'sha256': hashlib.sha256(pixel).hexdigest(), 'sourceSha256': hashlib.sha256(pixel).hexdigest(), 'locale': locale, 'reviewed': True}for identity in ('fixture-screen','fixture-screen-copy')}}))
            artifacts.append(build_pack(author, site / (locale + '-fixture.zip'), locale, 'fixture.1'))
        # Translation-only update: identical Chinese source/manual version, new
        # pack identity. It must replace mounted chapter text and image caches.
        for chapter in chapters:
            p=author/'locales/en'/(chapter['id']+'.json');catalog=read_json(p)
            for entry in catalog['entries'].values():
                if entry['source']=='基频为 100 Hz。':entry['translation']='Fundamental frequency remains 100 Hz.'
            p.write_bytes(dump(catalog))
        build_pack(author,site/'en-update.zip','en','fixture.2')
        build(author, site / 'manual')
        # Exercise the real distribution encoder. Foreign packs still refer to
        # original PNG/WAV identities; the host serves only packed WebP/MP3 paths.
        base=read_json(site/'manual/project.json');packed_root=site/'manual'
        base['assets']=[package_asset(copy.deepcopy(asset),author/asset['path'],packed_root)[0]for asset in base['assets']]
        (packed_root/'project.json').write_bytes(dump(base))
        (site/'bad-pack.zip').write_bytes(b'not a valid ZIP')
        (scratch / 'fixture.json').write_bytes(dump({'fixtureOnly': True, 'site': str(site), 'out': str(out), 'scratch': str(scratch), 'packs': artifacts}))
        result = subprocess.run(['node', str(ROOT / 'scripts/manual/verify_packs.cjs'), str(scratch / 'fixture.json')], cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError('Browser manual pack validation failed; see ' + str(out / 'report.json'))
        print(json.dumps({'report': str(out / 'report.json'), 'fixtureOnly': True, 'success': True}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
