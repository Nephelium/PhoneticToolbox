"""Schema-backed, standard-library manual validation and deterministic compilation."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO / 'frontend/src/manual/schema-chapter.json').read_text(encoding='utf-8'))
NODES = set(SCHEMA['x-supportedNodes'])
MARKS = set(SCHEMA['x-supportedMarks'])
ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$')
PRIVATE_PATH = re.compile(r'(?:[A-Za-z]:[\\/]|\\\\[^\\]|file://)')


class ManualError(ValueError):
    pass


def valid_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ManualError(f'{label}: 稳定 ID 无效')
    return value


def relative(value: Any, prefix: str | None = None) -> str:
    if (not isinstance(value, str) or not value or re.search(r'[\\:\x00-\x1f?#%]', value)
            or value.startswith('/') or any(x in ('', '.', '..') for x in value.split('/'))
            or (prefix and not value.startswith(prefix + '/'))):
        raise ManualError('路径必须为工程内的规范相对路径')
    return value


def safe_file(root: Path, value: str) -> Path:
    relative(value)
    current = root
    for part in value.split('/'):
        current = current / part
        if current.is_symlink():
            raise ManualError(f'工程内不允许符号链接: {value}')
    return current


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManualError(f'JSON 无法读取: {path.name} ({type(exc).__name__})') from exc


def is_public(asset: dict[str, Any]) -> bool:
    return asset.get('distribution', 'public') == 'public' and asset.get('git') is not False


def node_text(node: dict[str, Any]) -> str:
    if node.get('type') == 'text':
        return node.get('text', '')
    attrs = node.get('attrs', {})
    own = ' '.join(str(attrs[k]) for k in ('caption', 'label', 'latex') if attrs.get(k))
    return ' '.join(filter(None, [own, *[node_text(c) for c in node.get('content', [])]])).strip()


def validate(project_dir: Path, distribution: str = 'software', strict: bool = False) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    project = read_json(safe_file(project_dir, 'project.json'))
    if project.get('schemaVersion') != 'ptb-manual/1':
        raise ManualError('工程 schemaVersion 须为 ptb-manual/1')
    valid_id(project.get('id'), '工程')
    if not isinstance(project.get('title'), str) or not isinstance(project.get('chapters'), list) or not isinstance(project.get('assets'), list):
        raise ManualError('工程标题、章节和素材清单格式无效')
    warnings: list[str] = []
    errors: list[str] = []
    assets: dict[str, Any] = {}
    chapters: list[dict[str, Any]] = []
    chapter_ids: set[str] = set()
    paths: set[str] = set()
    references: set[str] = set()
    crossrefs: list[tuple[str, dict[str, Any]]] = []
    anchor_by_chapter: dict[str, set[str]] = {}
    skipped: list[str] = []
    media_bytes = 0
    for reference in project.get('references', []):
        key = valid_id(reference.get('id'), '参考文献')
        if key in references:
            errors.append(f'参考文献重复: {key}')
        references.add(key)
    for asset in project['assets']:
        key = valid_id(asset.get('id'), '素材')
        if key in assets:
            errors.append(f'素材 ID 重复: {key}')
        assets[key] = asset
        relative(asset.get('path'), 'assets')
        if asset.get('kind') not in ('image', 'audio', 'video', 'example'):
            errors.append(f'{key}: 素材类型无效')
        if asset.get('distribution') not in ('public', 'software-only'):
            errors.append(f'{key}: 素材分发范围无效')
        if asset.get('distribution') == 'software-only' and asset.get('git') is not False:
            errors.append(f'{key}: software-only 必须 git=false')
        if PRIVATE_PATH.search(json.dumps(asset.get('source', ''), ensure_ascii=False)):
            errors.append(f'{key}: source 包含私有绝对来源路径')
        if distribution == 'public' and not is_public(asset):
            skipped.append(key)
            continue
        file = safe_file(project_dir, asset['path'])
        if not file.is_file():
            errors.append(f'{key}: 缺少媒体 {asset["path"]}')
            continue
        data = file.read_bytes()
        media_bytes += len(data)
        if asset.get('sha256') and hashlib.sha256(data).hexdigest() != asset['sha256']:
            errors.append(f'{key}: 媒体 SHA-256 不一致')
    for descriptor in project['chapters']:
        chapter_id = valid_id(descriptor.get('id'), '章节')
        relative(descriptor.get('path'), 'chapters')
        if chapter_id in chapter_ids or descriptor['path'] in paths:
            errors.append(f'章节 ID 或路径重复: {chapter_id}')
        chapter_ids.add(chapter_id)
        paths.add(descriptor['path'])
        chapter = read_json(safe_file(project_dir, descriptor['path']))
        chapters.append(chapter)
        if chapter.get('schemaVersion') != 'ptb-manual-chapter/1' or chapter.get('id') != chapter_id or chapter.get('body', {}).get('type') != 'doc':
            errors.append(f'{chapter_id}: 章节版本、稳定 ID 或正文根节点无效')
            continue
        if chapter.get('title') != descriptor.get('title'):
            errors.append(f'{chapter_id}: 正文与目录标题不一致')
        anchors: set[str] = set()
        anchor_by_chapter[chapter_id] = anchors
        count = 0

        def walk(node: dict[str, Any], depth: int = 0) -> None:
            nonlocal count
            count += 1
            if count > 100000 or depth > 45:
                raise ManualError(f'{chapter_id}: 正文节点过多或过深')
            if not isinstance(node, dict) or not isinstance(node.get('type'), str):
                raise ManualError(f'{chapter_id}: 节点结构无效')
            kind, attrs = node['type'], node.get('attrs', {})
            if kind not in NODES:
                (errors if strict else warnings).append(f'{chapter_id}: 未支持节点 {kind}，保留原结构')
            if not isinstance(attrs, dict):
                raise ManualError(f'{chapter_id}: attrs 无效')
            if attrs.get('id'):
                anchor = valid_id(attrs['id'], f'{chapter_id} 锚点')
                if anchor in anchors:
                    errors.append(f'{chapter_id}: 锚点重复 {anchor}')
                anchors.add(anchor)
            if any(key.lower().startswith('on') or key in ('src', 'innerHTML', 'script') for key in attrs):
                errors.append(f'{chapter_id}: 不允许脚本或直接媒体地址')
            if kind in ('image', 'audio', 'video'):
                key = attrs.get('assetId')
                if key not in assets:
                    errors.append(f'{chapter_id}: 未登记媒体 {key}')
                elif distribution == 'public' and key in skipped:
                    warnings.append(f'{chapter_id}: 受限媒体 {key} 在公开版显示占位')
                elif assets[key].get('kind') != kind:
                    errors.append(f'{chapter_id}: 媒体类型不一致 {key}')
            if kind == 'crossReference':
                crossrefs.append((chapter_id, attrs))
            if kind == 'citation' and attrs.get('referenceId') not in references:
                errors.append(f'{chapter_id}: 未登记文献 {attrs.get("referenceId")}')
            if kind == 'heading' and not isinstance(attrs.get('level'), int):
                errors.append(f'{chapter_id}: 标题级别无效')
            if kind == 'text' and not isinstance(node.get('text'), str):
                errors.append(f'{chapter_id}: 文字节点无效')
            for mark in node.get('marks', []):
                if mark.get('type') not in MARKS:
                    (errors if strict else warnings).append(f'{chapter_id}: 未支持文字样式 {mark.get("type")}')
                href = mark.get('attrs', {}).get('href')
                if mark.get('type') == 'link' and (not isinstance(href, str) or not re.match(r'^(https?://|mailto:|#[A-Za-z0-9_.:-]+$|manual:)', href, re.I)):
                    errors.append(f'{chapter_id}: 链接协议不安全')
            for child in node.get('content', []):
                walk(child, depth + 1)

        walk(chapter['body'])
        if not chapter['body'].get('content'):
            warnings.append(f'{chapter_id}: 正文预留为空 ({descriptor.get("status", "draft")})')
        if descriptor.get('status') in ('draft', 'in_progress', 'deferred', 'planned'):
            warnings.append(f'{chapter_id}: 当前章为 {descriptor["status"]}，不标作已完成')
    for chapter_id, attrs in crossrefs:
        target_chapter = attrs.get('chapterId') or chapter_id
        redirects = project.get('redirects', {})
        target_chapter = redirects.get(target_chapter, target_chapter)
        if target_chapter not in chapter_ids:
            errors.append(f'{chapter_id}: 交叉引用目标章节不存在 {target_chapter}')
        elif attrs.get('targetId') and attrs['targetId'] not in anchor_by_chapter.get(target_chapter, set()):
            errors.append(f'{chapter_id}: 交叉引用目标锚点不存在 {attrs["targetId"]}')
    return {'project': project, 'chapters': chapters, 'warnings': list(dict.fromkeys(warnings)), 'errors': list(dict.fromkeys(errors)), 'skippedAssets': skipped, 'mediaBytes': media_bytes}


def reading_project(result: dict[str, Any], distribution: str) -> dict[str, Any]:
    project = copy.deepcopy(result['project'])
    if distribution == 'public':
        project['assets'] = [a for a in project['assets'] if is_public(a)]
    search = []
    for descriptor, chapter in zip(project['chapters'], result['chapters']):
        sections = []
        target = None
        for block in chapter['body'].get('content', []):
            if block.get('type') == 'heading':
                target = block.get('attrs', {}).get('id')
                if target:
                    sections.append({'id': target, 'title': node_text(block), 'level': block['attrs'].get('level', 2)})
            text = node_text(block)
            if text:
                entry = {'chapterId': chapter['id'], 'title': chapter['title'], 'text': text}
                if target:
                    entry['targetId'] = target
                search.append(entry)
        descriptor['sections'] = sections
    project['searchIndex'] = search
    project['distribution'] = distribution
    return project


def dump(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
