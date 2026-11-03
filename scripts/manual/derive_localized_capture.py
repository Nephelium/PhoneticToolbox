"""Derive a reviewed manual detail from actual locale-specific DOM regions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from verification_artifacts import verification_run
from manual.core import ManualError, dump, read_json, safe_file


def region_pixels(region: dict, dpr: float) -> list[int]:
    box = region.get('box')
    if region.get('units') != 'css-px' or not isinstance(region.get('selector'), str) or not region['selector'] or not isinstance(box, list) or len(box) != 4:
        raise ManualError('局部图需要当前语言实际 DOM selector 与 CSS-px box')
    if not all(type(v) in (int, float) and math.isfinite(v) for v in box) or box[2] <= 0 or box[3] <= 0:
        raise ManualError('局部图 DOM 范围无效')
    left, top, width, height = box
    return [math.floor(left * dpr), math.floor(top * dpr), math.ceil((left + width) * dpr), math.ceil((top + height) * dpr)]


def crop_pixels(region: dict, dpr: float, padding: float) -> list[int]:
    if type(padding) not in (int, float) or not math.isfinite(padding) or not 0 <= padding <= 100:
        raise ManualError('局部图留白须在 0–100 CSS 像素之间')
    left, top, right, bottom = region_pixels(region, dpr)
    pad = math.ceil(padding * dpr)
    crop = [max(0, left - pad), max(0, top - pad), min(2560, right + pad), min(1600, bottom + pad)]
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ManualError('局部图不在实际截图范围内')
    return crop


def selected_region(regions: dict, primary: str, included: list[str]) -> dict:
    selected = [regions[name] for name in [primary, *included]]
    for item in selected:
        region_pixels(item, 1)
    left = min(item['box'][0] for item in selected)
    top = min(item['box'][1] for item in selected)
    right = max(item['box'][0] + item['box'][2] for item in selected)
    bottom = max(item['box'][1] + item['box'][3] for item in selected)
    return {'selector': ','.join(item['selector'] for item in selected), 'units': 'css-px',
            'box': [left, top, right - left, bottom - top]}


def main(args):
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QImage, QPainter, QPen, QColor
    parent_path = args.parent_report.resolve(strict=True)
    if not parent_path.is_relative_to(ROOT / 'output/validation'):
        raise ManualError('父截图须来自当前任务验收目录')
    parent = read_json(parent_path)
    if parent.get('success') is not True or parent.get('sourceChangesDuringRun') != [] or parent.get('locale') not in ('en', 'ja'):
        raise ManualError('父截图报告未完成或来源发生改变')
    matches = [item for item in parent['captures'] if item['name'] == args.parent_capture]
    if len(matches) != 1:
        raise ManualError('父截图名必须唯一')
    capture = matches[0]
    regions = capture.get('regions', {})
    included = getattr(args, 'include', [])
    if args.region not in regions or any(name not in regions for name in [*args.mark, *included]):
        raise ManualError('所需局部/标注区域未由当前界面记录')
    image_path = safe_file(parent_path.parent, capture['file'])
    image = QImage(str(image_path))
    if image.isNull() or [image.width(), image.height()] != [2560, 1600] or hashlib.sha256(image_path.read_bytes()).hexdigest() != capture['sha256']:
        raise ManualError('父完整截图尺寸或摘要不符')
    crop = crop_pixels(selected_region(regions, args.region, included), capture['dpr'], args.padding)
    detail = image.copy(crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1])
    detail.setDevicePixelRatio(1)
    annotations = []
    if args.mark:
        painter = QPainter(detail)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor('#d62828'), 4))
        for name in args.mark:
            left, top, right, bottom = region_pixels(regions[name], capture['dpr'])
            relative = [left - crop[0], top - crop[1], right - crop[0], bottom - crop[1]]
            if not (0 <= relative[0] < relative[2] <= detail.width() and 0 <= relative[1] < relative[3] <= detail.height()):
                painter.end()
                raise ManualError('标注框不在派生范围内，应重新选择区域与留白')
            painter.drawRoundedRect(QRectF(relative[0], relative[1], relative[2] - relative[0], relative[3] - relative[1]), 10, 10)
            annotations.append({'region': name, 'box': relative, 'type': 'roundedRectangle', 'width': 4, 'radius': 10, 'color': '#d62828'})
        painter.end()
    with verification_run('manual-localized-details', parent['locale'], 'Actual locale-specific full screenshot crop from recorded DOM boxes; optional rounded rectangle annotations; no control/data replacement') as (out, scratch):
        target = out / 'detail.png'
        if not detail.save(str(target)):
            raise ManualError('无法保存局部图')
        derived = {'method': 'qt-crop-and-annotations/1', 'tool': str(Path(__file__).relative_to(ROOT).as_posix()),
                   'toolSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'parentReport': parent_path.relative_to(ROOT).as_posix(), 'parentReportSha256': hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                   'parentCapture': capture['name'], 'parentImageSha256': capture['sha256'],
                   'region': args.region, 'includeRegions': included, 'paddingCssPx': args.padding, 'crop': crop, 'annotations': annotations}
        entry = {key: capture[key] for key in ('locale', 'fontLocale', 'fontFamily', 'dpr', 'pageZoom', 'theme', 'maximized')}
        entry.update(name=args.asset_id, assetId=args.asset_id, file='detail.png', image=[detail.width(), detail.height()],
                     sha256=hashlib.sha256(target.read_bytes()).hexdigest(), derive=derived)
        sources = dict(parent['sources'])
        sources['deriveTool'] = {'path': str(Path(__file__)), 'sha256': derived['toolSha256']}
        report = {'success': True, 'locale': parent['locale'], 'sourceChapter': args.chapter,
                  'captures': [entry], 'sources': sources, 'sourceChangesDuringRun': parent['sourceChangesDuringRun']}
        if parent.get('fixtureOnly') is True:
            report['fixtureOnly'] = True
        (out / 'report.json').write_bytes(dump(report))
        print(json.dumps({'report': str(out / 'report.json'), 'image': entry['image'], 'assetId': args.asset_id}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--parent-capture', required=True)
    parser.add_argument('--chapter', required=True)
    parser.add_argument('--asset-id', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--mark', action='append', default=[])
    parser.add_argument('--include', action='append', default=[], help='Include another actual DOM region in the crop union')
    parser.add_argument('--padding', type=float, default=12)
    try:
        main(parser.parse_args())
    except (ManualError, OSError, ValueError, KeyError) as exc:
        raise SystemExit('派生局部图失败: ' + str(exc))
