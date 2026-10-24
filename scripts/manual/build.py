from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
if __package__:
    from .core import SCHEMA, ManualError, dump, is_public, read_json, reading_project, safe_file, validate
else:
    from core import SCHEMA, ManualError, dump, is_public, read_json, reading_project, safe_file, validate


def reader_chapter(chapter: dict) -> dict:
    """Tiptap's absent optional anchors become absent keys in reading JSON."""
    result=copy.deepcopy(chapter)
    def visit(node):
        attrs=node.get('attrs',{})
        if attrs.get('id',False) is None:
            del attrs['id']
        for child in node.get('content',[]):
            visit(child)
    visit(result['body'])
    return result


def referenced_asset_ids(chapters: list[dict]) -> set[str]:
    """Collect schema media references through every nested content container."""
    media_nodes = {kind for kind, attrs in SCHEMA['x-nodeAttrs'].items() if 'assetId' in attrs}
    referenced = set()
    pending = [chapter['body'] for chapter in chapters]
    while pending:
        node = pending.pop()
        if not isinstance(node, dict) or not isinstance(node.get('content', []), list):
            raise ManualError('节点结构无效')
        if node.get('type') in media_nodes:
            attrs = node.get('attrs', {})
            if not isinstance(attrs, dict) or not isinstance(attrs.get('assetId'), str):
                raise ManualError('媒体 assetId 无效')
            referenced.add(attrs['assetId'])
        pending.extend(node.get('content', []))
    return referenced


def validate_referenced_projection(project_dir: Path, parent: Path, distribution: str) -> tuple[dict, list[str]]:
    """Validate an isolated reader projection; never rewrite author metadata/media."""
    project = read_json(safe_file(project_dir, 'project.json'))
    if not isinstance(project, dict) or not isinstance(project.get('chapters'), list) or not isinstance(project.get('assets'), list):
        raise ManualError('工程标题、章节和素材清单格式无效')
    chapters = [read_json(safe_file(project_dir, descriptor['path'])) for descriptor in project['chapters']]
    referenced = referenced_asset_ids(chapters)
    unreferenced = [asset['id'] for asset in project['assets']
                    if (distribution != 'public' or is_public(asset)) and asset['id'] not in referenced]
    excluded = set(unreferenced)
    project['assets'] = [asset for asset in project['assets'] if asset['id'] not in excluded]
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.manual-reader-projection-', dir=parent) as temporary:
        projection = Path(temporary)
        (projection / 'project.json').write_bytes(dump(project))
        for descriptor, chapter in zip(project['chapters'], chapters):
            target = safe_file(projection, descriptor['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(dump(chapter))
        for asset in project['assets']:
            if distribution == 'public' and not is_public(asset):
                continue
            source = safe_file(project_dir, asset['path'])
            if source.is_file():
                target = safe_file(projection, asset['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        # Missing referenced files, hashes, node/metadata and path rules still
        # use the same strict validator as the default author build.
        result = validate(projection, distribution, strict=True)
    return result, unreferenced


def write_atomic(path: Path, data: bytes) -> None:
    # An open reader may hold media on Windows. Identical bytes are already
    # published, so leave that file and its handle untouched.
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(uuid.uuid4()))
    try:
        with temp.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError as exc:
            print(f'生成暂存清理失败，已保留 {temp}: {exc}', file=sys.stderr)


def retired_outputs(output: Path, include: set[str]) -> dict[str, str]:
    """Only retire files claimed by a previous complete reader manifest/report."""
    if not output.exists():
        return {}
    actual = set()
    pending = [output]
    while pending:
        for file in pending.pop().iterdir():
            if file.is_symlink() or getattr(file.lstat(), 'st_file_attributes', 0) & 0x400:
                raise ManualError('输出目录内存在联接或符号链接')
            if file.is_dir():
                pending.append(file)
            elif file.is_file():
                actual.add(file.relative_to(output).as_posix())
    extra = actual - include
    if not extra:
        return {}
    try:
        old_project = json.loads(safe_file(output, 'project.json').read_text('utf-8'))
        old_report = json.loads(safe_file(output, 'build-report.json').read_text('utf-8'))
        claimed = {a['path'] for a in old_project['assets']} | {c['path'] for c in old_project['chapters']}
        hashes = old_report['hashes']
        if (old_project['schemaVersion'] != 'ptb-manual/1' or old_report['schemaVersion'] != 'ptb-manual-build/1'
                or set(hashes) != claimed or not extra <= claimed):
            raise ValueError('unclaimed output')
        retired = {}
        for name in extra:
            if not name.startswith(('assets/', 'chapters/')) or not re.fullmatch('[a-f0-9]{64}', hashes[name]):
                raise ValueError('invalid old output identity')
            if hashlib.sha256(safe_file(output, name).read_bytes()).hexdigest() != hashes[name]:
                raise ValueError('changed old output')
            retired[name] = hashes[name]
        return retired
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManualError('输出目录含未登记或已变化的旧文件，保留并请先核查；不会自动删除。') from exc


def build(project_dir: Path, output: Path, distribution: str = 'software', *, referenced_assets_only: bool = False) -> dict:
    for ancestor in (output.absolute(), *output.absolute().parents):
        if ancestor.exists() and (ancestor.is_symlink() or getattr(ancestor.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ManualError('输出路径不允许联接或符号链接')
    project_dir, output = project_dir.resolve(), output.resolve()
    if output == project_dir or output in project_dir.parents or project_dir in output.parents and '.studio' not in output.parts:
        raise ManualError('构建目标不可覆盖源工程或源工程内普通目录')
    unreferenced = []
    if referenced_assets_only:
        result, unreferenced = validate_referenced_projection(project_dir, output.parent, distribution)
    else:
        result = validate(project_dir, distribution, strict=True)
    if result['errors']:
        raise ManualError('\n'.join(result['errors']))
    project = reading_project(result, distribution)
    include = {d['path'] for d in project['chapters']} | {a['path'] for a in project['assets']} | {'project.json', 'build-report.json'}
    if distribution == 'public' and output.exists():
        for existing in output.rglob('*'):
            if existing.is_symlink():
                raise ManualError('输出目录内存在符号链接')
            if existing.is_file() and existing.relative_to(output).as_posix() not in include:
                raise ManualError('公开输出目录含旧的非公开或未列入文件，请指定新的独立目录。构建不会自动删除旧文件。')
    retired = retired_outputs(output, include) if distribution == 'software' else {}
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for descriptor, chapter in zip(project['chapters'], result['chapters']):
        data = dump(reader_chapter(chapter))
        write_atomic(safe_file(output, descriptor['path']), data)
        hashes[descriptor['path']] = hashlib.sha256(data).hexdigest()
    selected_media_bytes = 0
    for asset in project['assets']:
        data = safe_file(project_dir, asset['path']).read_bytes()
        selected_media_bytes += len(data)
        write_atomic(safe_file(output, asset['path']), data)
        hashes[asset['path']] = hashlib.sha256(data).hexdigest()
    # Manifest is replaced last. A reader never sees a new manifest before its files exist.
    write_atomic(safe_file(output, 'project.json'), dump(project))
    report = {'schemaVersion': 'ptb-manual-build/1', 'distribution': distribution, 'chapters': len(project['chapters']), 'assets': len(project['assets']), 'mediaBytes': selected_media_bytes if referenced_assets_only else result['mediaBytes'], 'skippedAssets': result['skippedAssets'], 'warnings': result['warnings'], 'hashes': hashes}
    if referenced_assets_only:
        report.update(referencedAssetsOnly=True, unreferencedAssets=unreferenced)
    write_atomic(safe_file(output, 'build-report.json'), dump(report))
    # Retire only unchanged, previously generated files after publishing the new
    # reader. The author project and unknown files are never cleanup candidates.
    for name, digest in retired.items():
        stale = safe_file(output, name)
        if hashlib.sha256(stale.read_bytes()).hexdigest() != digest:
            raise ManualError('生成后旧文件发生变化，已保留: ' + name)
        stale.unlink()
    return report


def main() -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='从说明书源工程生成离线只读资源')
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distribution', choices=('software', 'public'), default='software')
    parser.add_argument('--referenced-assets-only', action='store_true', help='只在读本中包含章节实际引用的素材；不改动源工程')
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.project, args.output, args.distribution, referenced_assets_only=args.referenced_assets_only), ensure_ascii=False, indent=2))
        return 0
    except (ManualError, OSError) as exc:
        print(f'生成失败: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
