"""Prepare reviewable manual chapters from existing v3 documents; no media processing."""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MODULES = [
    ('M16', '录音', 'recording'), ('M05', '唇形提取', 'lip-extraction'),
    ('M01', '参数估计', 'parameter-estimation'), ('M02', '参数显示', 'parameter-display'),
    ('M03', 'EGG 信号分析', 'egg-analysis'), ('M04', 'LPC 谱图', 'lpc-spectrum'),
    ('M06', '声学参数合成', 'speech-synthesis'), ('M10', '生理参数合成', 'vocal-tract'),
    ('M07', '发声类型合成', 'phonation-synthesis'), ('M08', '变速变调', 'pitch-manipulation'),
    ('M09', '语谱图转音频', 'spectrogram-to-audio'), ('M17', '国际音标表Plus', 'ipa-plus'),
    ('M13', '汉字转国际音标', 'mandarin-ipa'), ('M11', 'MFA 自动标注', 'mfa'),
    ('M12', 'TextGrid标注', 'annotation'), ('M14', '音系归纳', 'phonology-induction'),
    ('M15', '感知实验', 'perception'),
]


def inline(value: str) -> list[dict]:
    result: list[dict] = []
    # Common existing Markdown marks; preserve all non-marked prose as text.
    pattern = r'(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^\)]+\)|\*[^*]+\*)'
    for part in re.split(pattern, value):
        if not part:
            continue
        mark = None
        if part.startswith('**') and part.endswith('**'):
            part, mark = part[2:-2], {'type': 'bold'}
        elif part.startswith('`') and part.endswith('`'):
            part, mark = part[1:-1], {'type': 'code'}
        elif part.startswith('*') and part.endswith('*'):
            part, mark = part[1:-1], {'type': 'italic'}
        else:
            match = re.fullmatch(r'\[([^\]]+)\]\(([^\)]+)\)', part)
            if match:
                part = match[1]
                # Current developer-document relative links are not public reading URLs.
                if match[2].startswith(('https://', 'http://')):
                    mark = {'type': 'link', 'attrs': {'href': match[2], 'target': '_blank'}}
        node = {'type': 'text', 'text': part}
        if mark:
            node['marks'] = [mark]
        result.append(node)
    return result


def paragraph(value: str, **attrs) -> dict:
    node = {'type': 'paragraph', 'content': inline(value)}
    if attrs:
        node['attrs'] = attrs
    return node


def markdown_nodes(text: str, chapter_id: str) -> tuple[list[dict], list[dict]]:
    lines = text.splitlines()
    nodes: list[dict] = []
    sections: list[dict] = []
    index = 0
    section_count = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith('```'):
            language = line[3:].strip()
            index += 1
            body = []
            while index < len(lines) and not lines[index].strip().startswith('```'):
                body.append(lines[index])
                index += 1
            nodes.append({'type': 'codeBlock', 'attrs': {'language': language},
                          'content': [{'type': 'text', 'text': '\n'.join(body)}]})
            index += 1
            continue
        heading = re.match(r'^(#{1,4})\s+(.+)', line)
        if heading:
            if len(heading[1]) > 1:
                section_count += 1
                anchor = f'{chapter_id}-section-{section_count:02d}'
                level = min(len(heading[1]), 4)
                nodes.append({'type': 'heading', 'attrs': {'id': anchor, 'level': level},
                              'content': inline(heading[2])})
                sections.append({'id': anchor, 'title': heading[2], 'level': level})
            index += 1
            continue
        if line.startswith('|') and index + 1 < len(lines) and re.match(r'^\s*\|?[\s:|\-]+\|\s*$', lines[index + 1]):
            rows = []
            row_index = 0
            while index < len(lines) and lines[index].strip().startswith('|'):
                current = lines[index].strip()
                if not re.fullmatch(r'[\s:|\-]+', current):
                    cells = current.strip('|').split('|')
                    rows.append({'type': 'tableRow', 'content': [
                        {'type': 'tableHeader' if row_index == 0 else 'tableCell',
                         'content': [paragraph(cell.strip())]} for cell in cells]})
                    row_index += 1
                index += 1
            nodes.append({'type': 'table', 'content': rows})
            continue
        listing = re.match(r'^(?:([-*])|\d+[.)])\s+(.+)', line)
        if listing:
            ordered = listing[1] is None
            items = []
            while index < len(lines):
                item = re.match(r'^(?:([-*])|\d+[.)])\s+(.+)', lines[index].strip())
                if not item or (item[1] is None) != ordered:
                    break
                items.append({'type': 'listItem', 'content': [paragraph(item[2])]})
                index += 1
            nodes.append({'type': 'orderedList' if ordered else 'bulletList', 'content': items})
            continue
        if line.startswith('>'):
            nodes.append({'type': 'blockquote', 'content': [paragraph(line.lstrip('> ').strip())]})
        elif re.fullmatch(r'[-*_]{3,}', line):
            nodes.append({'type': 'horizontalRule'})
        else:
            nodes.append(paragraph(line))
        index += 1
    return nodes, sections


class LegacyText(HTMLParser):
    """Keep old manual as source material in ignored evidence, never product instructions."""
    def __init__(self):
        super().__init__()
        self.chapter = 0
        self.heading = None
        self.skip = 0
        self.parts: dict[int, list[str]] = {}

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        if tag == 'h1' and not self.skip:
            self.chapter += 1
            self.parts[self.chapter] = []
        if tag in ('h1', 'h2', 'h3', 'h4') and self.chapter and not self.skip:
            self.parts[self.chapter].append('\n' + '#' * int(tag[1]) + ' ')
        elif tag in ('p', 'li', 'tr', 'pre') and self.chapter and not self.skip:
            self.parts[self.chapter].append('\n')
        if tag in ('img', 'audio') and self.chapter and not self.skip:
            src = dict(attrs).get('src', '')
            self.parts[self.chapter].append(f'\n[历史媒体 {tag}: {src}]\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)
        if tag in ('p', 'li', 'tr', 'pre', 'h1', 'h2', 'h3', 'h4') and self.chapter and not self.skip:
            self.parts[self.chapter].append('\n')

    def handle_data(self, data):
        if self.chapter and not self.skip:
            self.parts[self.chapter].append(data)


def save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT / 'manual')
    parser.add_argument('--legacy-source', type=Path)
    parser.add_argument('--legacy-output', type=Path, default=ROOT / 'output/manual-work/v2-reference')
    args = parser.parse_args()
    project_path = args.project / 'project.json'
    project = json.loads(project_path.read_text(encoding='utf-8')) if project_path.exists() else {
        'schemaVersion': 'ptb-manual/1', 'id': 'phonetic-toolbox-v3',
        'title': 'PhoneticToolbox 3.0 使用说明', 'language': 'zh-CN',
        'version': '2026.10.05-draft', 'chapters': [], 'assets': [], 'references': [],
    }
    existing = {c['id']: c for c in project['chapters']}
    descriptors = []
    for module_id, title, slug in MODULES:
        chapter_id = module_id.lower()
        relative = f'chapters/{chapter_id}.json'
        destination = args.project / relative
        if chapter_id in existing and destination.exists():
            descriptors.append(existing[chapter_id])
            continue
        if module_id == 'M10':
            nodes, sections, status = [], [], 'deferred'
        elif module_id == 'M15':
            nodes = [paragraph('本模块正在修改，定稿后更新操作说明与截图。')]
            sections, status = [], 'draft'
        else:
            source = ROOT / 'docs/manual' / (slug + '.md')
            nodes, sections = markdown_nodes(source.read_text(encoding='utf-8'), chapter_id)
            status = 'draft'
        save(destination, {'schemaVersion': 'ptb-manual-chapter/1', 'id': chapter_id,
                           'title': title, 'moduleId': module_id, 'body': {'type': 'doc', 'content': nodes}})
        descriptors.append({'id': chapter_id, 'moduleId': module_id, 'title': title,
                            'path': relative, 'status': status, 'sections': sections})
    extra = [c for c in project['chapters'] if c.get('moduleId') not in {x[0] for x in MODULES}]
    project['chapters'] = extra + descriptors
    save(project_path, project)
    if args.legacy_source:
        legacy = LegacyText()
        legacy.feed(args.legacy_source.read_text(encoding='utf-8'))
        args.legacy_output.mkdir(parents=True, exist_ok=True)
        for number, fragments in legacy.parts.items():
            (args.legacy_output / f'chapter-{number:02d}.md').write_text(
                ''.join(fragments).strip() + '\n', encoding='utf-8')
        save(args.legacy_output / 'source.json', {'sha256': hashlib.sha256(args.legacy_source.read_bytes()).hexdigest(),
                                                'chapters': len(legacy.parts), 'purpose': 'read-only historical source'})
    print(json.dumps({'project': str(project_path), 'chapters': len(project['chapters']),
                      'status': 'draft preparation, not final content verification'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
