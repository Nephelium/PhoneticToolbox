"""Check the editorial rules shared by all V2-style V3 manual chapters."""

from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / 'manual'


def visit(node: dict):
    yield node
    for child in node.get('content', []):
        yield from visit(child)


def main() -> None:
    project = json.loads((MANUAL / 'project.json').read_text(encoding='utf-8'))
    assets = {item['id']: item for item in project['assets']}
    issues: list[str] = []
    figures = audios = tables = 0
    for descriptor in project['chapters']:
        chapter_id = descriptor['id']
        chapter = json.loads((MANUAL / descriptor['path']).read_text(encoding='utf-8'))
        body = chapter['body']['content']
        if chapter_id == 'm10':
            if body:
                issues.append('m10: deferred chapter must remain empty')
            continue
        if not any(node['type'] == 'heading' and node.get('attrs', {}).get('level') == 2 for node in body):
            issues.append(chapter_id + ': no top-level section heading')
        for index, top in enumerate(body):
            for node in visit(top):
                kind = node['type']
                attrs = node.get('attrs', {})
                if kind == 'heading':
                    if attrs.get('level') not in (2, 3, 4, 5):
                        issues.append(chapter_id + ': heading level outside 2–5')
                    title = ''.join(n.get('text', '') for n in visit(node) if n['type'] == 'text')
                    if re.match(r'^\s*\d+(?:\.\d+)*[.、\s]+', title):
                        issues.append(chapter_id + ': manually numbered heading ' + title[:35])
                elif kind in ('image', 'audio', 'video'):
                    asset = assets.get(attrs.get('assetId'), {})
                    caption = attrs.get('caption') or asset.get('caption')
                    if not isinstance(caption, str) or not caption.strip():
                        issues.append(chapter_id + ': media without caption ' + str(attrs.get('assetId')))
                    if kind == 'image':
                        figures += 1
                        if not isinstance(attrs.get('alt'), str) or not attrs['alt'].strip():
                            issues.append(chapter_id + ': image without alternative text ' + str(attrs.get('assetId')))
                    elif kind == 'audio':
                        audios += 1
                        if index < 2:
                            issues.append(chapter_id + ': audio placed before explanatory text')
                elif kind == 'table':
                    tables += 1
                    if not isinstance(attrs.get('caption'), str) or not attrs['caption'].strip():
                        issues.append(chapter_id + ': table without caption')
                    rows = [n for n in node.get('content', []) if n['type'] == 'tableRow']
                    if not rows or not any(cell['type'] == 'tableHeader' for cell in rows[0].get('content', [])):
                        issues.append(chapter_id + ': table without header row')
    result = {'figures': figures, 'audios': audios, 'tables': tables, 'issues': issues}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
