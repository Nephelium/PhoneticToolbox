"""Attach complete approved locale notes before signing or publishing a manifest."""
from pathlib import Path
import argparse
import json


def attach_notes(manifest: dict, notes_dir: Path) -> dict:
    if manifest.get('schemaVersion') != 'ptb-release/1':
        raise ValueError('Expected ptb-release/1')
    if 'signature' in manifest:
        raise ValueError('Attach notes before signing; do not modify a signed manifest')
    version = manifest.get('version')
    if not isinstance(version, str) or not version or any(c not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-+' for c in version):
        raise ValueError('Invalid release version')
    notes = {}
    for language, suffix in [('zh-CN', ''), ('en', '.en'), ('ja', '.ja')]:
        text = (notes_dir / f'{version}{suffix}.md').read_text('utf-8')
        if not text.strip() or len(text) > 50000:
            raise ValueError(f'Empty or oversized {language} notes')
        notes[language] = text
    if manifest.get('notes', notes['zh-CN']) != notes['zh-CN']:
        raise ValueError('Existing notes differ from the approved Chinese notes')
    return {**manifest, 'notes': notes['zh-CN'], 'notesByLocale': notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--notes-dir', type=Path, default=Path(__file__).parent / 'notes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = attach_notes(json.loads(args.manifest.read_text('utf-8')), args.notes_dir)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
