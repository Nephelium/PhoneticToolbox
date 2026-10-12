"""Refresh the affected M14 figures and prose from a verified native capture."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = {
    'import': ('m14-figure-m14-import-detail-r4-light', '样例表占用剩余高度，确认导入并继续固定在底部。测试.xlsx 无表头，开始行选 1，三列分别是字头、IPA 与备注。'),
    'initial': ('m14-figure-m14-initial-preview-r5-light', '声母 → 韵母预览按 DOCX 显示居中标题和双栏统计表，代表例字与备注下标同序呈现，统计可折叠。'),
    'final': ('m14-figure-m14-final-preview-r5-light', '韵母 → 声母预览滚到正文，ã 分类居中，其下各声母按段连续排列。六个调类名称仅为操作暂拟。'),
    'matrix': ('m14-figure-m14-matrix-preview-r5-light', '二维表首组 12 行×10 列，列标题与行标题固定。长格可在格内滚动，顶部切换声母或韵母分组。'),
    'result': ('m14-figure-m14-result-detail-r4-light', '5453 条记录生成完整三文件，预览绑定本次设置与导出字体。预览、搜索、分页与右侧生成和保存入口留在当前工作区。'),
}
TEXT = {
    '选择文件后先查看样例，确认导入成功才替换当前数据。': '选择文件后先查看样例，确认导入成功才替换当前数据。样例表随窗口高度收缩，确认导入并继续保持在工作区底部；其余样例在表格内部滚动查看。文本编码与分隔符在左侧导入设置中选择。',
    '按声母或韵母作为外层查看同一批记录，展开声母、韵母与声调统计核对类别和顺序。': '按声母或韵母作为外层查看同一批记录。预览沿用 DOCX 的居中标题、声母／韵母／声调双栏统计表与代表例字，每类选输入顺序中前五个不同的字头和备注组合。统计默认展开，可折叠后查看正文。正文外层分类只显示一次，各声韵组合按段连续排列，调类标签与备注下标保留。',
    '两种分组表每页 100 条，本例共 55 页；分组可能跨页。': '两种分组表每批 100 条，本例共 55 批；页码用于屏幕浏览，分组可能跨批。翻页按钮固定在预览下方，不必滚到正文末尾。',
    '行是韵母，列是声母。每组最多 12 行×10 列，组内通过横向、纵向滚动查看。': '行是韵母，列是声母。每组最多 12 行×10 列，组内通过横向、纵向滚动查看，顶部列标题与左侧行标题固定。长格在格内滚动，每格每批最多 60 条，其余条目用格内翻页按钮查看。界面分批呈现行，切换后稍候即可看到完整当前分组；导出保留全部条目。',
    '页面预览用来核对内容与顺序，Word 的纸面分页请在实际文件中检查。XLSX 列宽固定，行高按内容估计；长格若显示不全，可在编辑栏查看全文或调整布局。': '两份 DOCX 预览按 Word 的统计表、分类标题、正文与备注下标组织，字体采用当前导出设置；生成后固定为该次字体快照。屏幕每批 100 条与 Word 实际纸张分页不同，纸面分页仍在实际文件中检查。二维预览通过表内和格内滚动浏览长内容。XLSX 列宽固定，行高按内容估计；长格若显示不全，可在编辑栏查看全文或调整布局。',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve(strict=True)
    report = json.loads(report_path.read_text('utf-8'))
    assert report['success']
    project_path, chapter_path = ROOT / 'manual/project.json', ROOT / 'manual/chapters/m14.json'
    project = json.loads(project_path.read_text('utf-8'))
    chapter = json.loads(chapter_path.read_text('utf-8'))
    for name, (figure_id, caption) in FIGURES.items():
        capture = next(c for c in report['captures'] if c['name'] == name)
        assert capture['image'] == [2560, 1600] and capture['dpr'] == 1.5
        assert capture['maximized'] and capture['pageZoom'] == 1 and capture['theme'] == 'light'
        source = Path(capture['file'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == capture['sha256']
        asset_id = 'm14-r2-' + name + '-' + digest[:8]
        relative = 'assets/software-only/m14-r2/' + asset_id + '.png'
        target = ROOT / 'manual' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        if not any(a['id'] == asset_id for a in project['assets']):
            project['assets'].append(dict(id=asset_id, path=relative, kind='image', mime='image/png',
                sha256=digest, width=2560, height=1600, caption=caption, git=False, distribution='software-only',
                sourceType='实际应用截图', source='Windows 当前源码 Qt，工程内现有调查字表，原生 DPR 1.5、页面 100%、浅色、最大化。调类仅作暂拟示例。',
                capture={'report': report_path.relative_to(ROOT).as_posix(), **{k: v for k, v in capture.items() if k != 'file'}}))
        figure = next(n for n in chapter['body']['content'] if n['type'] == 'image' and n['attrs']['id'] == figure_id)
        figure['attrs'].update(assetId=asset_id, caption=caption, alt=caption)
    replaced = set()

    def revise(node):
        if isinstance(node, dict):
            if node.get('type') == 'text' and node.get('text') in TEXT:
                replaced.add(node['text'])
                node['text'] = TEXT[node['text']]
            for value in node.values():
                revise(value)
        elif isinstance(node, list):
            for item in node:
                revise(item)

    revise(chapter)
    assert len(replaced) == len(TEXT), 'Expected prose changed since this registration was prepared'
    for path, value in ((project_path, project), (chapter_path, chapter)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'figures': len(FIGURES), 'prose_updates': len(replaced)}))


if __name__ == '__main__':
    main()
