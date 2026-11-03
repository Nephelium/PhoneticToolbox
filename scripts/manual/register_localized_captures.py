"""Register visually reviewed, locale-proven actual captures without changing Chinese.

Only image IDs referenced by the selected chapter can be registered. Existing
other-language/chapter entries and the authoritative Chinese project are kept.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import struct
import sys
import os
import time

if __package__:
    from .build import referenced_asset_ids, write_atomic
    from .core import ManualError, dump, read_json, safe_file
else:
    from build import referenced_asset_ids, write_atomic
    from core import ManualError, dump, read_json, safe_file

ROOT = Path(__file__).resolve().parents[2]


def validate_scientific(capture: dict, asset: dict, project_dir: Path, dimensions: tuple[int, int]):
    proof = capture.get('scientificExport')
    reviews = read_json(safe_file(project_dir, 'locales/scientific-figures.json'))
    review = reviews.get('figures', {}).get(asset['id'], {})
    source_hash = hashlib.sha256(safe_file(project_dir, asset['path']).read_bytes()).hexdigest()
    if reviews.get('schemaVersion') != 'ptb-manual-scientific-review/1' or review.get('reviewed') is not True or review.get('decision') != 'localize-generated-labels' or review.get('sourceSha256') != source_hash:
        raise ManualError('只有已核实需要本地化生成标签的科学图可登记导出')
    if not isinstance(proof, dict) or proof.get('schemaVersion') != 'ptb-manual-scientific-export/1' or proof.get('sourceAssetId') != asset['id'] or proof.get('sourceImageSha256') != source_hash:
        raise ManualError('科学图导出缺少对应当前中文源图的证据')
    if list(dimensions) != capture.get('image') or not (0 < dimensions[0] <= 20000 and 0 < dimensions[1] <= 20000 and dimensions[0] * dimensions[1] <= 40_000_000):
        raise ManualError('科学导出实际图片尺寸无效')
    baseline = proof.get('baseline')
    if not isinstance(baseline, dict) or not isinstance(baseline.get('evidence'), str) or not baseline['evidence'].strip() or not isinstance(proof.get('action'), str) or not proof['action'].strip():
        raise ManualError('科学图导出须给出已核实旧配方与真实按钮动作')
    for field in ('input', 'parameters'):
        if not isinstance(proof.get(field), dict) or not isinstance(baseline.get(field), dict):
            raise ManualError('科学导出须保留完整输入和参数 JSON')
        digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest(proof[field]) != proof.get(field + 'Sha256') or digest(baseline[field]) != proof.get(field + 'Sha256'):
            raise ManualError('科学导出与已核实原配方的输入或参数不一致: ' + field)


def validate_derived(capture: dict, dimensions: tuple[int, int], locale: str):
    if __package__:
        from .derive_localized_capture import crop_pixels, region_pixels, selected_region
    else:
        from derive_localized_capture import crop_pixels, region_pixels, selected_region
    proof = capture.get('derive')
    tool = 'scripts/manual/derive_localized_capture.py'
    if not isinstance(proof, dict) or proof.get('method') != 'qt-crop-and-annotations/1' or proof.get('tool') != tool or proof.get('toolSha256') != hashlib.sha256(safe_file(ROOT, tool).read_bytes()).hexdigest():
        raise ManualError('局部图缺少当前派生工具及版本证据')
    parent_path = safe_file(ROOT, proof.get('parentReport', ''))
    if not parent_path.is_relative_to(ROOT / 'output/validation') or hashlib.sha256(parent_path.read_bytes()).hexdigest() != proof.get('parentReportSha256'):
        raise ManualError('局部图父报告范围或摘要不符')
    parent = read_json(parent_path)
    if parent.get('success') is not True or parent.get('sourceChangesDuringRun') != [] or parent.get('locale') != locale:
        raise ManualError('局部图父报告未完整通过')
    originals = [item for item in parent.get('captures', []) if item.get('name') == proof.get('parentCapture')]
    if len(originals) != 1:
        raise ManualError('局部图父截图不唯一')
    original = originals[0]
    raw = safe_file(parent_path.parent, original['file']).read_bytes()
    if raw[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', raw[16:24]) != (2560, 1600) or hashlib.sha256(raw).hexdigest() != proof.get('parentImageSha256') or original.get('sha256') != proof.get('parentImageSha256'):
        raise ManualError('局部图父完整图片尺寸或摘要不符')
    if any(original.get(key) != capture.get(key) for key in ('locale', 'fontLocale', 'dpr', 'pageZoom', 'theme', 'maximized')):
        raise ManualError('局部图与父截图的语言或显示状态不符')
    regions = original.get('regions', {})
    included = proof.get('includeRegions', [])
    if not isinstance(included, list) or proof.get('region') not in regions or any(name not in regions for name in included):
        raise ManualError('局部图未使用父截图实际 DOM 区域')
    crop = crop_pixels(selected_region(regions, proof['region'], included), original['dpr'], proof.get('paddingCssPx'))
    if crop != proof.get('crop') or dimensions != (crop[2] - crop[0], crop[3] - crop[1]) or list(dimensions) != capture.get('image'):
        raise ManualError('局部图裁剪框与实际尺寸不符')
    for annotation in proof.get('annotations', []):
        if annotation.get('region') not in regions:
            raise ManualError('局部图标注未使用实际 DOM 区域')
        box = region_pixels(regions[annotation['region']], original['dpr'])
        box = [box[0] - crop[0], box[1] - crop[1], box[2] - crop[0], box[3] - crop[1]]
        if annotation != {'region': annotation['region'], 'box': box, 'type': 'roundedRectangle', 'width': 4, 'radius': 10, 'color': '#d62828'}:
            raise ManualError('局部图标注证据与实际 DOM 范围不符')


@contextmanager
def catalogue_lock(project_dir: Path, locale: str):
    """Serialize independent capture agents without keeping locks in sources."""
    identity = hashlib.sha256(str(project_dir.resolve()).encode()).hexdigest()[:16]
    path = safe_file(ROOT, f'output/validation/manual-language-packs/registration-locks/{identity}-{locale}.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        acquired = False
        deadline = time.monotonic() + 20
        while not acquired:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise ManualError('其他截图正在登记，请稍后重试')
                time.sleep(.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def validate_window(capture: dict, chapter_id: str, asset_id: str) -> bool:
    fullscreen = capture.get('fullscreen') is True
    if fullscreen:
        if chapter_id != 'm18' or asset_id != 'm18-r3-reader-fullscreen-da8aa864' or capture.get('maximized') is not False or any(capture.get(key) is not True for key in ('documentFullscreen', 'wasMaximized', 'fullscreenExitRestoredMaximized')):
            raise ManualError('全屏阅读截图须提供原生与DOM全屏、此前最大化及退出后恢复的真实证据')
    elif capture.get('maximized') is not True:
        raise ManualError('普通界面截图须由实际最大化窗口采集')
    return fullscreen


def register(project_dir: Path, report_file: Path, locale: str, chapter_id: str, *, reviewed: bool = False, names: tuple[str, ...] = (), asset_map: dict[str, str] | None = None, scientific_export: bool = False) -> dict:
    if locale not in ('en', 'ja') or not reviewed:
        raise ManualError('须先逐张目视复核，再指定 --reviewed 与 en/ja')
    if ROOT not in report_file.parents:
        raise ManualError('采图报告须来自当前工程的独立验收目录')
    report = read_json(report_file)
    if report.get('success') is not True or report.get('fixtureOnly') is True or report.get('locale') != locale or report.get('sourceChapter') != chapter_id or report.get('sourceChangesDuringRun') != []:
        raise ManualError('采图报告未完整通过、语言不符或源在运行中改变')
    for source in report.get('sources', {}).values():
        file = Path(source['path'])
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != source['sha256']:
            raise ManualError('采图源已变化，须核对后重新采集: ' + str(file))
    project = read_json(safe_file(project_dir, 'project.json'))
    descriptor = next((c for c in project['chapters'] if c['id'] == chapter_id), None)
    if descriptor is None:
        raise ManualError('目标章节不存在')
    chapter = read_json(safe_file(project_dir, descriptor['path']))
    selected = referenced_asset_ids([chapter])
    assets = [a for a in project['assets'] if a['id'] in selected and a['kind'] == 'image']
    catalog_path = safe_file(project_dir, f'locales/{locale}/assets.json')
    catalog = read_json(catalog_path) if catalog_path.exists() else {'schemaVersion': 'ptb-manual-localized-assets/1', 'locale': locale, 'overrides': {}}
    if catalog.get('schemaVersion') != 'ptb-manual-localized-assets/1' or catalog.get('locale') != locale:
        raise ManualError('已有本地化截图登记格式无效')
    additions, writes = {}, []
    captures = report.get('captures', [])
    asset_map = asset_map or {}
    if set(asset_map) - {item.get('name') for item in captures}:
        raise ManualError('显式素材映射的截图不在报告中')
    if names and set(names) - {item.get('name') for item in captures}:
        raise ManualError('指定的目视复核截图不在报告中')
    for capture in captures:
        if capture.get('captureKind') == 'scientific-export' and not scientific_export:
            if names and capture.get('name') in names:
                raise ManualError('科学导出图须单独显式 --scientific-export 登记')
            continue
        if capture.get('auxiliary') is True:
            if names and capture.get('name') in names:
                raise ManualError('辅助完整图须先派生局部图，不能直接登记为正式素材')
            continue
        if names and capture.get('name') not in names:
            continue
        if capture.get('locale') != locale or capture.get('fontLocale') != locale or capture.get('theme') != 'light' or capture.get('dpr') != 1.5 or capture.get('pageZoom') != 1:
            raise ManualError('截图语言、字体、DPR、缩放或主题证据不符')
        explicit = asset_map.get(capture.get('name')) or (capture.get('assetId') if capture.get('derive') else None) or (capture.get('scientificExport', {}).get('sourceAssetId') if capture.get('captureKind') == 'scientific-export' else None)
        matches = [a for a in assets if (a['id'] == explicit if explicit else capture.get('name') in (a.get('capture', {}).get('name'), a.get('capture', {}).get('originalId')))]
        if len(matches) != 1:
            raise ManualError('截图名不能唯一对应章节资产: ' + str(capture.get('name')))
        asset = matches[0]
        fullscreen = validate_window(capture, chapter_id, asset['id'])
        if asset['id'] in additions:
            raise ManualError('同一资产重复采集')
        source = safe_file(report_file.parent, capture['file'])
        data = source.read_bytes()
        if data[:8] != b'\x89PNG\r\n\x1a\n' or hashlib.sha256(data).hexdigest() != capture.get('sha256'):
            raise ManualError('实际截图尺寸或字节摘要不符: ' + source.name)
        dimensions = struct.unpack('>II', data[16:24])
        if capture.get('captureKind') == 'scientific-export':
            validate_scientific(capture, asset, project_dir, dimensions)
        elif capture.get('derive'):
            validate_derived(capture, dimensions, locale)
        elif dimensions != (2560, 1600):
            raise ManualError('完整截图须为 2560×1600，局部图须带可核验派生证据')
        prefix = 'assets/public' if asset['distribution'] == 'public' else 'assets/software-only'
        target = f'{prefix}/localized/{locale}/{asset["id"]}.png'
        original = safe_file(project_dir, asset['path'])
        additions[asset['id']] = {'path': target, 'sha256': capture['sha256'], 'sourceSha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                                  'locale': locale, 'reviewed': True, 'captureReport': report_file.relative_to(ROOT).as_posix(),
                                  'captureName': capture['name'], 'image': list(dimensions), 'dpr': 1.5, 'pageZoom': 1, 'theme': 'light',
                                  'fontLocale': locale, 'fontFamily': capture.get('fontFamily')}
        if capture.get('derive'):
            additions[asset['id']]['derive'] = capture['derive']
        if capture.get('captureKind') == 'scientific-export':
            additions[asset['id']]['scientificExport'] = capture['scientificExport']
        if fullscreen:
            additions[asset['id']]['fullscreen'] = {key: capture[key] for key in ('fullscreen', 'documentFullscreen', 'wasMaximized', 'fullscreenExitRestoredMaximized')}
        writes.append((safe_file(project_dir, target), data))
    if not additions:
        raise ManualError('报告未包含截图')
    with catalogue_lock(project_dir, locale):
        # Another module may have committed its captures during validation.
        # Reload inside the same lock that protects the atomic replacement.
        catalog = read_json(catalog_path) if catalog_path.exists() else {'schemaVersion': 'ptb-manual-localized-assets/1', 'locale': locale, 'overrides': {}}
        if catalog.get('schemaVersion') != 'ptb-manual-localized-assets/1' or catalog.get('locale') != locale or not isinstance(catalog.get('overrides'), dict):
            raise ManualError('已有本地化截图登记格式无效')
        for target, data in writes:
            write_atomic(target, data)
        catalog['overrides'].update(additions)
        write_atomic(catalog_path, dump(catalog))
    return {'locale': locale, 'chapter': chapter_id, 'registered': sorted(additions), 'count': len(additions),
            'bytes': sum(safe_file(project_dir, entry['path']).stat().st_size for entry in additions.values()),
            'sourceProjectUnchanged': True, 'report': report_file.relative_to(ROOT).as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT / 'manual')
    parser.add_argument('--locale', choices=('en', 'ja'), required=True)
    parser.add_argument('--chapter', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reviewed', action='store_true')
    parser.add_argument('--capture', action='append', default=[], help='Only individually reviewed capture names; repeat for multiple images')
    parser.add_argument('--asset', action='append', default=[], metavar='NAME=ID', help='Explicit reviewed mapping for older source assets without capture names')
    parser.add_argument('--scientific-export', action='store_true', help='Explicitly register verified same-input scientific exports, preserving their values')
    args = parser.parse_args()
    try:
        mappings = [item.split('=', 1) for item in args.asset]
        if any(len(item) != 2 or not all(item) for item in mappings) or len({item[0] for item in mappings}) != len(mappings):
            raise ManualError('显式素材映射须为不重复的 NAME=ID')
        result = register(args.project.resolve(), args.report.resolve(), args.locale, args.chapter, reviewed=args.reviewed, names=tuple(args.capture), asset_map=dict(mappings), scientific_export=args.scientific_export)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ManualError, OSError, ValueError, KeyError) as exc:
        print('截图登记失败: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
