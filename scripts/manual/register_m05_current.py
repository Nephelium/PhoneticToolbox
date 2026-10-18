"""Register verified M05 screenshots and update only the affected chapter/index."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from core import node_text

ROOT = Path(__file__).resolve().parents[2]
FIGURES = {
    'm05-preview-ready': ('m05-figure-m05-overview-r2-light', '预览模式的准备界面。左侧设置设备与参数，音频列表尚未开放时可点击授权并刷新设备。中央显示网格和曲线，右侧查看画面与历史。图中尚未开始采集。'),
    'm05-realtime-ready': ('m05-figure-m05-realtime-idle-current-light', '实时计算模式的准备界面。开始录制后同时采集音视频并计算候选唇形，另存时可选择视频与动画。'),
    'm05-video-recording-ready': ('m05-figure-m05-record-then-analyze-idle-current-light', '视频录制模式的准备界面。示例请求120 fps，录制只保存音视频，唇形稍后通过离线分析计算。请求值与实际设备帧率分别核对。'),
    'm05-filter-off-ready': ('m05-figure-m05-filter-off-idle-current-light', '预览模式关闭特征点防抖。截止频率保留15 Hz并禁用，麦克风可在预览前授权并预选。'),
    'm05-filter-detail': ('m05-filter-detail-figure', '关闭特征点防抖后，截止频率输入禁用。重新勾选防抖后可修改截止频率。'),
    'm05-methods-software': ('m05-figure-m05-methods-software-current-light', '方法与引用窗口的软件与代码来源页，可查看依赖版本、许可和引用。'),
    'm05-offline-ready': ('m05-offline-entry-figure', '离线分析入口位于顶部模式菜单。点击选择文件后，再点击分析所选文件。左侧显示输入文件，右侧保留回放与历史入口。'),
    'm05-offline-replay': ('m05-offline-replay-figure', '合成示例视频完成逐帧分析后的回放界面。左侧显示解码、推理与检出帧数，中央显示唇形网格，右侧显示原画面与同步回放控件。'),
    'm05-capture-statistics': ('m05-capture-statistics-figure', '视频录制结束后切到离线分析，左下角继续保留本页采集统计。展开统计含义可查看五个计数的口径。本例只录制媒体，已推理和检出均为0。'),
}


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '-', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve(strict=True)
    report_path.relative_to(ROOT / 'output/validation/m05-manual')
    report = json.loads(report_path.read_text('utf8'))
    assert report['success'] and not report['errors'] and not report['terminations']
    captures = {c['name']: c for c in report['captures']}
    assert set(captures) == set(FIGURES)
    manual = ROOT / 'manual'
    project = json.loads((manual / 'project.json').read_text('utf8'))
    chapter = json.loads((manual / 'chapters/m05.json').read_text('utf8'))
    blocks = chapter['body']['content']
    nodes = {n.get('attrs', {}).get('id'): n for n in blocks}
    asset_ids = {}
    for name, (figure_id, caption) in FIGURES.items():
        capture = captures[name]
        assert capture['dpr'] == 1.5 and capture['pageZoom'] == 1 and capture['maximized'] and capture['theme'] == 'light' and capture['navigationCollapsed']
        assert capture['image'] == [2560, 1600] or name == 'm05-filter-detail'
        data = Path(capture['file']).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == capture['sha256']
        asset_id = name + '-20261018-' + digest[:8]
        asset_ids[name] = asset_id
        relative = 'assets/software-only/m05-current-20261018/' + asset_id + '.png'
        target = manual / relative
        if not target.exists():
            atomic(target, data)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        asset = dict(id=asset_id, path=relative, kind='image', mime='image/png', sha256=digest,
                     width=capture['image'][0], height=capture['image'][1], caption=caption,
                     git=False, distribution='software-only', sourceType='实际应用截图局部标注' if name == 'm05-filter-detail' else '实际应用截图',
                     source='Windows 当前源码 Qt；浅色、最大化、原生DPR 1.5、页面100%。结果使用既有公开NASA测试图生成的合成视频，采集状态使用合成媒体输入，未使用实体摄像头或麦克风。',
                     capture={'report': report_path.relative_to(ROOT).as_posix(), **{k: v for k, v in capture.items() if k != 'file'}})
        if name == 'm05-filter-detail':
            original = captures['m05-filter-off-ready']
            asset.update(originalAssetId=asset_ids['m05-filter-off-ready'], originalSha256=original['sha256'], crop=capture['crop'], annotations=capture['annotations'])
        if not any(a['id'] == asset_id for a in project['assets']):
            project['assets'].append(asset)
        node = nodes.get(figure_id)
        if node is None:
            node = dict(type='image', attrs=dict(id=figure_id, width='100%', align='center'))
            nodes[figure_id] = node
        node['attrs'].update(assetId=asset_id, caption=caption, alt=caption)
    placements = [
        ('m05-offline-entry-figure', 'm05-controls-analysis'),
        ('m05-capture-statistics-figure', 'm05-replay'),
        ('m05-offline-replay-figure', 'm05-high-framerate-replay-note'),
    ]
    for figure_id, following_id in placements:
        if nodes[figure_id] not in blocks:
            index = next(i for i, n in enumerate(blocks) if n.get('attrs', {}).get('id') == following_id)
            blocks.insert(index, nodes[figure_id])
    for block in blocks:
        if block['type'] == 'paragraph':
            for child in block.get('content', []):
                if child.get('type') == 'text':
                    child['text'] = child['text'].replace('本章部分配图保留早期的实时预览、实时录制及高帧率先录后算名称，分别对应前三种模式。', '')
                    if child['text'].startswith('以下屏幕录制完整展示') and '录屏中的实时预览' not in child['text']:
                        child['text'] += '录屏中的实时预览对应当前的预览模式。'
    descriptor = next(c for c in project['chapters'] if c['id'] == 'm05')
    search, sections, target_id = [], [], None
    for block in blocks:
        if block['type'] == 'heading':
            target_id = block['attrs']['id']
            sections.append(dict(id=target_id, title=node_text(block), level=block['attrs'].get('level', 2)))
        text = node_text(block)
        if text:
            entry = dict(chapterId='m05', title=chapter['title'], text=text)
            if target_id:
                entry['targetId'] = target_id
            search.append(entry)
    descriptor['sections'] = sections
    old_index = project['searchIndex']
    first = next(i for i, entry in enumerate(old_index) if entry['chapterId'] == 'm05')
    project['searchIndex'] = [e for e in old_index[:first] if e['chapterId'] != 'm05'] + search + [e for e in old_index[first:] if e['chapterId'] != 'm05']
    for path, value in [(manual / 'chapters/m05.json', chapter), (manual / 'project.json', project)]:
        atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf8'))
    print(json.dumps(dict(figures=len(FIGURES), source_assets=len(project['assets']), chapter_images=sum(n['type'] == 'image' for n in blocks)), ensure_ascii=False))


if __name__ == '__main__':
    main()
