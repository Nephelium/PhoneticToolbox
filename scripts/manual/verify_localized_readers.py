"""Read all translated chapters back and audit invariant scientific/link/media data.

This checks structural fidelity, not whether every instruction matches current
product behavior. It deliberately does not build incomplete language packs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from verification_artifacts import verification_run
from scripts.manual.build import reader_chapter, referenced_asset_ids
from scripts.manual.core import dump, read_json
from scripts.manual.localization import VISIBLE_ATTRS, coverage, localized_reader


def invariant(node):
    """Ignore translated text while retaining scientific and object identity.

    Inline objects may move within their paragraph for natural target-language
    syntax. Their complete payload and multiplicity must remain unchanged.
    """
    objects, protected = Counter(), Counter()

    def walk(value):
        kind = value['type']
        if kind == 'codeBlock':
            protected[dump(value).decode()] += 1
            return
        if kind != 'text':
            attrs = {key: item for key, item in value.get('attrs', {}).items()
                     if key not in VISIBLE_ATTRS or kind == 'citation'}
            objects[json.dumps([kind, attrs], ensure_ascii=False, sort_keys=True)] += 1
        else:
            marks = value.get('marks', [])
            # Mark payload identifies links, code and IPA without translating
            # any protected scientific symbol or changing its destination.
            for mark in marks:
                objects[json.dumps(['mark', mark], ensure_ascii=False, sort_keys=True)] += 1
                if mark['type'] in ('code', 'ipa'):
                    protected[json.dumps([value['text'], mark], ensure_ascii=False, sort_keys=True)] += 1
        for child in value.get('content', []):
            walk(child)
    walk(node)
    return objects, protected


def main():
    project_dir = ROOT / 'manual'
    source = read_json(project_dir / 'project.json')
    originals = [reader_chapter(read_json(project_dir / item['path'])) for item in source['chapters']]
    expected_ids = [chapter['id'] for chapter in originals]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [project_dir / 'project.json', *(project_dir / item['path'] for item in source['chapters'])]}
    with verification_run('manual-language-packs', 'complete-readers',
                          'All translated chapters round-trip through localized_reader; stable IDs, formulas, links, media, table objects, code and IPA; semantic/current-feature review remains separate') as (out, scratch):
        report = {'success': False, 'scope': 'Full text reader construction and structural fidelity only',
                  'semanticCurrentFeatureAcceptance': False, 'locales': {}, 'sources': before}
        all_coverage = coverage(project_dir)
        for locale in ('en', 'ja'):
            assert all_coverage['locales'][locale]['complete'], locale + ': incomplete text'
            project, chapters, digest = localized_reader(project_dir, locale)
            assert [item['id'] for item in chapters] == expected_ids
            assert [item['id'] for item in project['chapters']] == expected_ids
            assert project['id'] == source['id'] and project['language'] == locale
            assert referenced_asset_ids(chapters) == referenced_asset_ids(originals)
            assert project.get('references', []) == source.get('references', [])
            details = []
            folder = scratch / locale
            folder.mkdir()
            (folder / 'project.json').write_bytes(dump(project))
            for descriptor, original, chapter in zip(project['chapters'], originals, chapters):
                assert invariant(original['body']) == invariant(chapter['body']), locale + '/' + chapter['id']
                # Actual serialized bytes must parse back identically.
                target = folder / descriptor['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = dump(chapter)
                target.write_bytes(payload)
                assert read_json(target) == chapter
                details.append({'chapter': chapter['id'], 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                                'objects': sum(invariant(chapter['body'])[0].values()), 'protectedFragments': sum(invariant(chapter['body'])[1].values())})
            report['locales'][locale] = {'sourceDigest': digest, 'textUnits': all_coverage['locales'][locale]['total'],
                                        'chapters': details, 'referencedAssets': len(project['assets']),
                                        'projectBytes': len(dump(project))}
        client = subprocess.run(['node', str(ROOT / 'scripts/manual/verify_localized_readers.mjs'), str(scratch)],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', check=True)
        report['clientReadback'] = json.loads(client.stdout)
        report['sourceChangesDuringRun'] = [path for path, digest in before.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
        assert not report['sourceChangesDuringRun']
        report['success'] = True
        (out / 'report.json').write_bytes(dump(report))
        print(json.dumps({'success': True, 'report': str(out / 'report.json'),
                          'locales': {locale: {'chapters': len(item['chapters']), 'textUnits': item['textUnits']} for locale, item in report['locales'].items()}}))


if __name__ == '__main__':
    main()
