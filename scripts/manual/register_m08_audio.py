"""Register actual M08 exported examples from a verified local capture report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / 'manual'
NAMES = (
    ('m08-freehand-demo', '手绘目标 F0 后由 V3 合成的短音节', {'method': 'freehand-f0'}),
    ('m08-peak-180', '拐点组合：0.32 秒峰值 180 Hz 的处理结果', {'method': 'breakpoint-combination', 'startHz': 140, 'peakHz': 180, 'endHz': 150, 'peakSeconds': 0.32}),
    ('m08-peak-220', '拐点组合：0.32 秒峰值 220 Hz 的处理结果', {'method': 'breakpoint-combination', 'startHz': 140, 'peakHz': 220, 'endHz': 150, 'peakSeconds': 0.32}),
    ('m08-peak-260', '拐点组合：0.32 秒峰值 260 Hz 的处理结果', {'method': 'breakpoint-combination', 'startHz': 140, 'peakHz': 260, 'endHz': 150, 'peakSeconds': 0.32}),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    if report.get('success') is not True or len(report.get('saved', [])) != len(NAMES):
        raise ValueError('Expected a successful four-file V3 export report')
    project_path = MANUAL / 'project.json'
    project = json.loads(project_path.read_text(encoding='utf-8'))
    assets = {row['id']: row for row in project['assets']}
    registered = []
    for saved, (asset_id, caption, config) in zip(report['saved'], NAMES):
        source = Path(report['out']) / 'm08-saved' / saved['name']
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != saved['sha256'] or len(raw) != saved['bytes']:
            raise ValueError('Export differs from the capture report: ' + saved['name'])
        info = sf.info(str(source))
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != 'PCM_16' or info.frames <= 0:
            raise ValueError('Unexpected exported audio format: ' + saved['name'])
        relative = f'assets/software-only/{asset_id}.wav'
        target = MANUAL / relative
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('Manual audio ID already has different content: ' + asset_id)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        row = dict(id=asset_id, path=relative, kind='audio', mime='audio/wav', sha256=digest,
                   sourceType='V3 实际处理音频', source='作者许可短音节在当前 M08 工作台处理并导出；见独立截图与保存报告',
                   caption=caption, distribution='software-only', git=False, config=config,
                   duration=info.duration, sampleRate=info.samplerate, channels=info.channels)
        if asset_id in assets and assets[asset_id]['sha256'] != digest:
            raise ValueError('Manual asset metadata collision: ' + asset_id)
        assets[asset_id] = row
        registered.append({'id': asset_id, 'sha256': digest, 'duration': info.duration})
    project['assets'] = list(assets.values())
    project_path.write_text(json.dumps(project, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'registered': registered}, ensure_ascii=False))


if __name__ == '__main__':
    main()
