"""Source-bound, paragraph-level manual translations and optional offline packs.

The Chinese author project remains the only structural source. Catalogs contain
whole paragraphs with movable XML markers for formatted spans/inline objects.
No translation service is called by authoring, building, or reading.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from html import escape
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

if __package__:
    from .core import ManualError, dump, read_json, reading_project, safe_file, validate
    from .build import reader_chapter, referenced_asset_ids, write_atomic
else:
    from core import ManualError, dump, read_json, reading_project, safe_file, validate
    from build import reader_chapter, referenced_asset_ids, write_atomic

LOCALES = tuple(item['code'] for item in read_json(Path(__file__).resolve().parents[2] / 'frontend/src/i18n/languages.json') if item['code'] != 'zh-CN')
CATALOG_SCHEMA = 'ptb-manual-localization/1'
PACK_SCHEMA = 'ptb-manual-language-pack/1'
HAN = re.compile(r'[\u3400-\u9fff]')
VISIBLE_ATTRS = ('caption', 'alt', 'note', 'title', 'label')


def numeric_tokens(text: str) -> Counter:
    # Unit-equivalent writing is allowed (4000 万像素 = 40 million pixels).
    # Ordinary values keep their original spelling/precision.
    values = []
    for match in re.finditer(r'(\d+(?:\.\d+)?)(?:\s*(百万|billion\b|million\b|thousand\b|亿|億|万|千))?', text, flags=re.IGNORECASE):
        number, scale = match.groups()
        if scale:
            multiplier = {'万': 10000, '千': 1000, '百万': 1000000, '亿': 100000000, '億': 100000000,
                          'billion': 1000000000, 'million': 1000000, 'thousand': 1000}[scale.lower()]
            values.append('scaled:' + format((Decimal(number) * multiplier).normalize(), 'f'))
        else:
            values.append(number)
    return Counter(values)


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


@dataclass
class Unit:
    owner: dict
    field: str
    source: str
    markers: list[dict] | None = None

    @property
    def key(self) -> str:
        # Positions and paragraph IDs deliberately do not participate. Inserting
        # another paragraph cannot invalidate an unchanged translation.
        return fingerprint({'source': self.source, 'markers': self.markers})

    def apply(self, translation: str) -> None:
        if self.markers is None:
            self.owner[self.field] = translation
        else:
            self.owner[self.field] = translated_inline(translation, self.markers)


def inline_source(content: list[dict]) -> tuple[str, list[dict]]:
    parts, markers = [], []
    for node in content:
        if node.get('type') == 'text' and not node.get('marks'):
            parts.append(escape(node.get('text', ''), quote=False))
        else:
            index = len(markers)
            markers.append(copy.deepcopy(node))
            if node.get('type') == 'text':
                parts.append(f'<s id="{index}">{escape(node.get("text", ""), quote=False)}</s>')
            else:
                parts.append(f'<x id="{index}"/>')
    return ''.join(parts), markers


def translated_inline(translation: str, markers: list[dict]) -> list[dict]:
    # Only a bounded, entity-free XML fragment is accepted. No arbitrary HTML,
    # attributes, links, nodes, or code can be introduced through a catalog.
    if len(translation) > 200000 or '<!' in translation or '<?' in translation:
        raise ManualError('译文包含不允许的 XML 声明或长度超限')
    try:
        root = ET.fromstring('<translation>' + translation + '</translation>')
    except ET.ParseError as exc:
        raise ManualError('译文内联标记无效: ' + str(exc)) from exc
    result, seen = [], set()

    def append_text(value: str | None):
        if value:
            result.append({'type': 'text', 'text': value})

    append_text(root.text)
    for item in root:
        if item.tag not in ('s', 'x') or set(item.attrib) != {'id'} or len(item):
            raise ManualError('译文只能使用源段落的 s/x 内联标记')
        identity = item.attrib['id']
        if not re.fullmatch(r'0|[1-9][0-9]*', identity):
            raise ManualError('译文内联标记 ID 无效')
        index = int(identity)
        if index >= len(markers) or index in seen:
            raise ManualError('译文内联标记缺失、重复或越界')
        node = copy.deepcopy(markers[index])
        if (node.get('type') == 'text') != (item.tag == 's'):
            raise ManualError('译文内联标记类型与原文不一致')
        if item.tag == 's':
            if not item.text:
                raise ManualError('带格式译文不能为空')
            if any(mark.get('type') in ('code', 'ipa') for mark in node.get('marks', [])) and item.text != node.get('text'):
                raise ManualError('代码、参数键和 IPA 标记文字须保留原文')
            node['text'] = item.text
        elif item.text:
            raise ManualError('引用和公式等内联对象不能改写内容')
        result.append(node)
        seen.add(index)
        append_text(item.tail)
    if seen != set(range(len(markers))):
        raise ManualError('译文遗漏原文的格式、链接或内联对象')
    return result


def string_unit(owner: dict, field: str) -> list[Unit]:
    value = owner.get(field)
    return [Unit(owner, field, value)] if isinstance(value, str) and HAN.search(value) else []


def chapter_units(chapter: dict) -> list[Unit]:
    units = string_unit(chapter, 'title') + string_unit(chapter, 'summary')

    def visit(node: dict):
        kind = node.get('type')
        # Code, original bibliographic labels, equations and asset identities
        # belong to the source. Captions around them remain translatable.
        if kind == 'codeBlock':
            return
        if kind != 'citation':
            for field in VISIBLE_ATTRS:
                units.extend(string_unit(node.get('attrs', {}), field))
        content = node.get('content', [])
        if kind in ('paragraph', 'heading') and content:
            source, markers = inline_source(content)
            if HAN.search(source):
                units.append(Unit(node, 'content', source, markers))
            return
        if kind == 'text':
            units.extend(string_unit(node, 'text'))
        for child in content:
            visit(child)

    visit(chapter['body'])
    return units


def project_units(project: dict, referenced: set[str]) -> list[Unit]:
    units = string_unit(project, 'title')
    for asset in project.get('assets', []):
        if asset['id'] not in referenced:
            continue
        for field in ('title', 'caption', 'alt', 'sourceType'):
            units.extend(string_unit(asset, field))
    return units


def groups(project: dict, chapters: list[dict]) -> dict[str, list[Unit]]:
    return {'project': project_units(project, referenced_asset_ids(chapters)), **{c['id']: chapter_units(c) for c in chapters}}


def asset_inventory(project_dir: Path) -> dict:
    """Metadata classification is a review queue, never a claim of visual QA."""
    project = read_json(safe_file(project_dir, 'project.json'))
    chapters = [read_json(safe_file(project_dir, d['path'])) for d in project['chapters']]
    used = referenced_asset_ids(chapters)
    references: dict[str, list[str]] = {}
    for chapter in chapters:
        for identity in referenced_asset_ids([chapter]):
            references.setdefault(identity, []).append(chapter['id'])
    assets = []
    for asset in project['assets']:
        source_type, kind = asset.get('sourceType', ''), asset['kind']
        category = 'ui-screenshot' if kind == 'image' and ('截图' in source_type or '屏幕' in source_type) else 'scientific-export' if kind == 'image' and ('导出' in source_type or '分布示意' in source_type) else kind
        file = safe_file(project_dir, asset['path'])
        assets.append({'id': asset['id'], 'path': asset['path'], 'kind': kind, 'sourceType': source_type,
                       'category': category, 'classification': 'metadata-only-needs-visual-review',
                       'referenced': asset['id'] in used, 'chapters': references.get(asset['id'], []),
                       'bytes': file.stat().st_size if file.is_file() else None,
                       'sha256': hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None,
                       'distribution': asset['distribution'], 'git': asset.get('git', True)})
    return {'schemaVersion': 'ptb-manual-localization-assets/1', 'assets': assets,
            'counts': {scope: {kind: {'count': sum(1 for a in assets if a['category'] == kind and (scope == 'all' or a['referenced'])),
                                    'bytes': sum(a['bytes'] or 0 for a in assets if a['category'] == kind and (scope == 'all' or a['referenced']))}
                               for kind in sorted({a['category'] for a in assets})} for scope in ('all', 'referenced')}}


def load_catalog(project_dir: Path, locale: str, group: str) -> dict:
    path = safe_file(project_dir, f'locales/{locale}/{group}.json')
    entries = {}
    if path.exists():
        data = read_json(path)
        if data.get('schemaVersion') != CATALOG_SCHEMA or data.get('locale') != locale or not isinstance(data.get('entries'), dict):
            raise ManualError('本地化目录格式无效: ' + str(path))
        entries = data['entries']
    if group == 'project':
        # Reuse a chapter caption only for the *same asset ID and exact source
        # caption*. Unrelated short labels never inherit another context's text.
        project = read_json(safe_file(project_dir, 'project.json'))
        assets = {a['id']: a for a in project['assets']}
        derived = {}
        for descriptor in project['chapters']:
            chapter = read_json(safe_file(project_dir, descriptor['path']))
            catalog = load_catalog(project_dir, locale, chapter['id'])
            pending = [chapter['body']]
            while pending:
                node = pending.pop()
                attrs = node.get('attrs', {})
                asset = assets.get(attrs.get('assetId'))
                if asset:
                    for field in ('caption', 'alt', 'title'):
                        source = asset.get(field)
                        if isinstance(source, str) and source in (attrs.get('caption'), attrs.get('alt'), attrs.get('title')):
                            key = Unit(asset, field, source).key
                            if key in catalog:
                                derived[key] = catalog[key]
                pending.extend(node.get('content', []))
        entries = {**derived, **entries}
    return entries


def apply_catalog(units: list[Unit], entries: dict, *, apply: bool = False) -> dict:
    unique = {unit.key: unit for unit in units}
    missing, invalid, translated = [], [], set()
    for key, unit in unique.items():
        entry = entries.get(key)
        if not entry or not isinstance(entry.get('translation'), str) or not entry['translation'].strip():
            missing.append({'key': key, 'source': unit.source})
            continue
        if entry.get('source') != unit.source:
            invalid.append({'key': key, 'error': '记录原文与当前原文不一致'})
            continue
        try:
            if unit.markers is not None:
                translated_inline(entry['translation'], unit.markers)
                source_text = ''.join(ET.fromstring('<r>' + unit.source + '</r>').itertext())
                translated_text = ''.join(ET.fromstring('<r>' + entry['translation'] + '</r>').itertext())
            else:
                source_text, translated_text = unit.source, entry['translation']
            if numeric_tokens(source_text) != numeric_tokens(translated_text):
                raise ManualError('译文须逐项保留原文的数字、默认值和参数范围')
        except ManualError as exc:
            invalid.append({'key': key, 'error': str(exc)})
            continue
        translated.add(key)
    if apply:
        if missing or invalid:
            raise ManualError(f'译文不完整或已过期: 缺失 {len(missing)}，无效 {len(invalid)}')
        for unit in units:
            unit.apply(entries[unit.key]['translation'])
    return {'total': len(unique), 'translated': len(translated), 'missing': missing, 'invalid': invalid,
            'obsolete': sorted(set(entries) - set(unique))}


def coverage(project_dir: Path, locales: tuple[str, ...] = LOCALES) -> dict:
    project = read_json(safe_file(project_dir, 'project.json'))
    chapters = [read_json(safe_file(project_dir, d['path'])) for d in project['chapters']]
    result = {'schemaVersion': 'ptb-manual-localization-coverage/1', 'sourceDigest': fingerprint({'project': project, 'chapters': chapters}), 'locales': {}}
    for locale in locales:
        reports = {name: apply_catalog(units, load_catalog(project_dir, locale, name)) for name, units in groups(project, chapters).items()}
        result['locales'][locale] = {'groups': reports, 'total': sum(r['total'] for r in reports.values()),
                                    'translated': sum(r['translated'] for r in reports.values()),
                                    'complete': all(not r['missing'] and not r['invalid'] for r in reports.values())}
    return result


def localized_reader(project_dir: Path, locale: str, distribution: str = 'software') -> tuple[dict, list[dict], str]:
    if locale not in LOCALES:
        raise ManualError('不支持的说明书语言: ' + locale)
    result = validate(project_dir, distribution, strict=True)
    if result['errors']:
        raise ManualError('\n'.join(result['errors']))
    source_digest = fingerprint({'project': result['project'], 'chapters': result['chapters']})
    chapters = [reader_chapter(c) for c in result['chapters']]
    project = copy.deepcopy(result['project'])
    referenced = referenced_asset_ids(chapters)
    project['assets'] = [a for a in project['assets'] if a['id'] in referenced]
    for name, units in groups(project, chapters).items():
        try:
            apply_catalog(units, load_catalog(project_dir, locale, name), apply=True)
        except ManualError as exc:
            raise ManualError(f'{locale}/{name}: {exc}') from exc
    for descriptor, chapter in zip(project['chapters'], chapters):
        descriptor['title'] = chapter['title']
        if 'summary' in chapter:
            descriptor['summary'] = chapter['summary']
    project['language'] = locale
    project = reading_project({**result, 'project': project, 'chapters': chapters}, distribution)
    project['sourceDigest'] = source_digest
    return project, chapters, source_digest


def build_pack(project_dir: Path, output: Path, locale: str, version: str, distribution: str = 'software') -> dict:
    """A ZIP holds text and per-language overrides; shared media stay in Chinese base."""
    project, chapters, source_digest = localized_reader(project_dir, locale, distribution)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', version):
        raise ManualError('语言包版本无效')
    payload = {'project.json': dump(project)}
    payload.update({d['path']: dump(c) for d, c in zip(project['chapters'], chapters)})
    overrides_path = safe_file(project_dir, f'locales/{locale}/assets.json')
    overrides_document = read_json(overrides_path) if overrides_path.exists() else {'schemaVersion': 'ptb-manual-localized-assets/1', 'locale': locale, 'overrides': {}}
    if overrides_document.get('schemaVersion') != 'ptb-manual-localized-assets/1' or overrides_document.get('locale') != locale or not isinstance(overrides_document.get('overrides'), dict):
        raise ManualError('本地化截图登记无效')
    overrides = overrides_document['overrides']
    # sourceType is already translated at this point, so classify against the
    # authoritative inventory rather than localized words.
    original_assets = read_json(safe_file(project_dir, 'project.json'))['assets']
    selected_ids = {a['id'] for a in project['assets']}
    required = {a['id'] for a in original_assets if a['id'] in selected_ids and a['kind'] == 'image' and any(word in a.get('sourceType', '') for word in ('截图', '屏幕'))}
    scientific = [a for a in original_assets if a['id'] in selected_ids and a['kind'] == 'image' and not any(word in a.get('sourceType', '') for word in ('截图', '屏幕'))]
    review_path = safe_file(project_dir, 'locales/scientific-figures.json')
    reviews = read_json(review_path) if review_path.exists() else {'schemaVersion': 'ptb-manual-scientific-review/1', 'figures': {}}
    if reviews.get('schemaVersion') != 'ptb-manual-scientific-review/1' or not isinstance(reviews.get('figures'), dict):
        raise ManualError('科学图本地化复核登记无效')
    for asset in scientific:
        review = reviews['figures'].get(asset['id'], {})
        actual = hashlib.sha256(safe_file(project_dir, asset['path']).read_bytes()).hexdigest()
        if review.get('sourceSha256') != actual or review.get('reviewed') is not True or review.get('decision') not in ('reuse-source', 'localize-generated-labels'):
            raise ManualError('科学图尚未按当前字节复核: ' + asset['id'])
        if review['decision'] == 'localize-generated-labels':
            required.add(asset['id'])
    if required - set(overrides):
        raise ManualError(f'{locale} 尚缺 {len(required - set(overrides))} 张当前正文引用的本地化界面截图或科学导出图')
    asset_overrides = {}
    by_id = {a['id']: a for a in project['assets']}
    for identity, item in overrides.items():
        asset = by_id.get(identity)
        if not asset or asset['kind'] != 'image' or not isinstance(item, dict):
            raise ManualError('本地化截图 ID 未引用或类型无效: ' + identity)
        path = item.get('path', '')
        if distribution == 'public' and (asset['distribution'] != 'public' or not path.startswith('assets/public/')):
            raise ManualError('公开语言包不得包含受限截图: ' + identity)
        if not path.startswith(('assets/public/', 'assets/software-only/')) or not path.endswith(('.png', '.webp')):
            raise ManualError('本地化截图须登记在正式素材目录: ' + identity)
        data = safe_file(project_dir, path).read_bytes()
        if item.get('sha256') != hashlib.sha256(data).hexdigest() or item.get('locale') != locale:
            raise ManualError('本地化截图摘要或语言登记不符: ' + identity)
        original = next(a for a in original_assets if a['id'] == identity)
        if item.get('sourceSha256') != hashlib.sha256(safe_file(project_dir, original['path']).read_bytes()).hexdigest():
            raise ManualError('本地化截图对应的中文源图已变化: ' + identity)
        if item.get('reviewed') is not True:
            raise ManualError('本地化截图尚未复核: ' + identity)
        # The digest is measured above, not guessed from asset names. Stable
        # IDs still map independently to their verified Chinese source bytes.
        name = 'assets/' + item['sha256'] + Path(path).suffix
        payload[name] = data
        asset_overrides[identity] = name
    manifest = {'schemaVersion': PACK_SCHEMA, 'locale': locale, 'version': version,
                'sourceDigest': source_digest, 'projectId': project['id'],
                'softwareVersion': project.get('softwareVersion'), 'manualVersion': project.get('version'),
                'distribution': distribution, 'sharedAssets': [a['id'] for a in project['assets']],
                'assetOverrides': asset_overrides, 'hashes': {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}}
    payload['language-pack.json'] = dump(manifest)
    # Refuse source-tree or existing-file targets. A failed or old build remains
    # inspectable; packaging never overwrites the author project.
    output = output.absolute()
    source = project_dir.resolve()
    if source == output or source in output.parents or output.exists():
        raise ManualError('语言包目标须为源工程以外的新文件')
    for ancestor in (output.parent, *output.parent.parents):
        if ancestor.exists() and (ancestor.is_symlink() or getattr(ancestor.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ManualError('语言包输出路径不允许联接或符号链接')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
    return {'locale': locale, 'version': version, 'sourceDigest': source_digest, 'file': output.name,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'bytes': output.stat().st_size,
            'chapters': len(chapters), 'files': len(payload), 'distribution': distribution}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('manual'))
    parser.add_argument('--locale', choices=LOCALES, action='append')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--inventory', action='store_true', help='按元数据列出实际引用截图与共用媒体，仍需目视复核')
    parser.add_argument('--pack', type=Path)
    parser.add_argument('--version')
    parser.add_argument('--distribution', choices=('software', 'public'), default='software')
    args = parser.parse_args()
    locales = tuple(args.locale or LOCALES)
    try:
        if args.inventory:
            report = asset_inventory(args.project)
        elif args.pack:
            if len(locales) != 1 or not args.version:
                raise ManualError('打包须指定唯一 --locale 和 --version')
            report = build_pack(args.project, args.pack, locales[0], args.version, args.distribution)
        else:
            report = coverage(args.project, locales)
        if args.report:
            write_atomic(args.report, dump(report))
        if 'locales' in report:
            print(json.dumps({k: {a: v for a, v in value.items() if a != 'groups'} for k, value in report['locales'].items()}, ensure_ascii=False, indent=2))
            return 1 if args.require_complete and any(not r['complete'] for r in report['locales'].values()) else 0
        print(json.dumps(report['counts'] if args.inventory else report, ensure_ascii=False, indent=2))
        return 0
    except (ManualError, OSError, ValueError) as exc:
        print('说明书本地化失败: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
