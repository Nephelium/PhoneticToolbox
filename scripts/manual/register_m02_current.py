"""Update M02's current controls, stable figure nodes and chapter search index."""
import argparse
import hashlib
import json
from pathlib import Path

from core import node_text
from register_m05_current import atomic

ROOT = Path(__file__).resolve().parents[2]


def paragraph(node, text):
    node['content'] = [dict(type='text', text=text)]


def steps(node, texts):
    items = node['content']
    for index, text in enumerate(texts):
        if index < len(items):
            paragraph(items[index]['content'][0], text)
            items[index]['content'] = items[index]['content'][:1]
        else:
            items.append(dict(type='listItem', content=[dict(type='paragraph', content=[dict(type='text', text=text)])]))
    node['content'] = items[:len(texts)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve(strict=True)
    report_path.relative_to(ROOT / 'output/validation/m02')
    report = json.loads(report_path.read_text('utf8'))
    assert report['success'] and not report['errors']
    chapter_path = ROOT / 'manual/chapters/m02.json'
    project_path = ROOT / 'manual/project.json'
    chapter = json.loads(chapter_path.read_text('utf8'))
    project = json.loads(project_path.read_text('utf8'))
    blocks = chapter['body']['content']
    nodes = {n.get('attrs', {}).get('id'): n for n in blocks}
    paragraphs = {
        'paragraph-17bd5b32-9cbe-4d66-b6e8-6720c7acdfc7': '下图已关联 XLSX 并勾选两项 F0，两条轨迹立即显示在当前图窗。右侧勾选始终对应当前图窗的内容。',
        'paragraph-828f9913-fef9-4ef9-ba33-86fa33294e0d': '数值列画曲线，文字列显示标签。Time_s、Source_Time_s 用于时间，不在候选中。点击图窗标题或图面，右侧会切换到该图窗的参数选择。',
        'paragraph-bf751281-85dd-4895-8bd8-5911d872e730': '右栏按宽度显示两至四列，可在候选区内滚动。搜索和筛选保留当前图窗已有的勾选；清空勾选会移除该窗的全部参数，包括暂时隐藏的项。',
        'paragraph-c7e79177-f406-4236-8d39-a20e8b882224': '放大图窗会扩大当前面板并隐藏其他参数窗，右侧参数仍可操作。切换当前图窗可直接查看另一窗，点击还原图窗恢复多窗。查看时间细节请缩短时间窗。',
        'paragraph-8c634113-1f69-4dbf-a8cd-00447979058d': 'SVG 保留矢量曲线和可编辑文字，内嵌 Doulos SIL 及 OFL；其他字体需对方设备支持。图例紧凑排列，优先放在绘图区空白位置，空间不足时移到图上方。长参数名称完整换行，双轴标明所属纵轴。分享固定外观可选 PNG。',
        'paragraph-0ce25282-5dea-4cb2-9f29-0118f7113727': '保存绘图配置记住图窗、标题和各窗的参数选择。重新载入源文件后，曲线与当前窗勾选一起恢复。文件、目录权限、时间窗、选区、搜索、放大状态和图片格式不随本机绘图配置保存；网页版设置随账号保留。',
        'paragraph-82268abd-2b26-4334-b7ed-1ffe937142b9': '下图恢复了 F1、SHR，图窗曲线与右侧勾选一致。视野和选区回到全长，其他未保存显示选项需重新核对。',
        'paragraph-60273fb4-15b9-4ade-95a7-75e9848cb38d': '清除筛选会清空搜索，并勾选 reaper、correction，曲线和原勾选保留。点击新建图窗会直接切换到空选择，无需先清空上一窗。',
        'paragraph-308fab50-4d40-4d60-8c29-3c7c305cbb12': '新视野读取失败时，图上可能暂留旧轨迹。等新视野完成后再解释和导图。重选表后保留新表中存在的参数，右栏与当前图窗一致；刷新或换音频回到全长。',
    }
    for ident, text in paragraphs.items():
        paragraph(nodes[ident], text)
    lists = {
        'orderedList-4bb40a8a-d9ec-466d-a922-7df75a85a8ff': ['确认当前图窗为图窗 1，勾选后即出现曲线和图例。取消任一勾选，对应曲线立即移除。'],
        'orderedList-57c08600-af2b-4511-8a8d-8dd3a91e8d3e': ['点击要编辑的图窗，或从右侧当前图窗下拉框选择。被选中的窗显示当前标记，右栏恢复它已有的勾选。', '搜索并勾选参数，曲线或文字立即显示；取消勾选立即移除。', '搜索和筛选只改变候选列表。全选可见参数追加当前可见列，已选隐藏项仍保留。', '清空勾选移除当前图窗全部数值曲线和文字层，保留空窗。其他图窗不受影响。', '各图窗分别保留自己的选择，同一参数可以在多个图窗中对照。'],
        'orderedList-2b5b53ec-c775-45e7-86a2-e45e4d808841': ['在图窗 1 勾选同类参数，例如两种 F0。', '点击新建图窗，当前窗立即切到新窗，右侧默认没有勾选。', '直接勾选另一组参数，各窗共用时间窗和选区，纵轴独立。点击原图窗可继续编辑原来的参数。'],
        'orderedList-9f33c76f-4967-4dfe-9d3a-f835ae9475c2': ['要在另一窗显示同一参数，切到该窗后直接勾选。原窗的曲线保持。', '合并并保留参数：在后续图窗点击合并回首图，所有参数并入第一窗，同名项只保留一次。随后核对纵轴。', '留下空窗用清空勾选；连窗一起去掉用删除图窗。最后一窗也可删除，再点新建图窗重新开始。', '布局满意后保存绘图配置，关闭前留意未保存提示。'],
    }
    for ident, texts in lists.items():
        steps(nodes[ident], texts)

    controls = nodes['m02-table-parameters']
    removed = {'清空选定图窗', '将 N 项分配到图窗', '从图中移除勾选项'}
    controls['content'] = [r for r in controls['content'] if node_text(r['content'][0]) not in removed]
    edits = {
        '参数复选框': ['参数复选框', '右栏候选区／开关', '新窗全不勾选', '勾选即在当前图窗绘制，取消即移除。切换图窗恢复对应勾选，长名称悬停查看。'],
        '清空勾选': ['清空勾选', '右栏／动作', '当前窗为空时禁用', '清掉当前图窗全部参数，包括筛选隐藏项，曲线和文字一起移除，图窗保留。'],
        '目标图窗': ['当前图窗', '右栏／图窗下拉框', '第一个已保存图窗，首次为图窗 1', '切换当前编辑图窗并恢复它的勾选。也可点击图窗标题或图面。'],
        '新建图窗': ['新建图窗', '右栏／动作', '首次已有一个空图窗', '新增空图窗并立即选中，右侧默认空选择，原图窗保持。最后一窗删除后也可重建。'],
        '删除选定图窗': ['删除图窗', '右栏及图窗内／动作', '有图窗时可用', '删除对应图窗及其参数选择，不转移曲线。当前窗删除后切回第一个保留图窗。'],
    }
    for row in controls['content']:
        values = edits.get(node_text(row['content'][0]))
        if values:
            for cell, text in zip(row['content'], values):
                paragraph(cell['content'][0], text)

    replacements = {
        '右侧：搜索、筛选、勾选参数，选择': '右侧：选择当前图窗，搜索、筛选并勾选参数；',
        '后分配。': '勾选立即绘制，取消立即移除。',
        '目标图窗': '当前图窗',
        '需要文字时，清空搜索，勾选文字列再分配。标签会出现在该图及上方波形。': '需要文字时，清空搜索并勾选文字列，标签立即出现在当前图及上方波形。',
        '重读会': '重读后',
        '，仅保留新表中仍存在的曲线分配。': '，并同步右侧勾选。',
        '选择正确结果或未关联参数表。重读后': '选择正确结果或未关联参数表。重读后',
        '勾选一层或多层文字，选目标并分配。图上方出现文字及区间边界。': '选择当前图窗并勾选一层或多层文字，图上方立即出现文字及区间边界。',
        '勾选后无曲线：先分配，确认目标有效、当前窗有有效数值。文字列只显示标签。': '勾选后无曲线：确认当前图窗正确、表已读完，当前时间窗有有效数值。文字列只显示标签。',
        '，再勾选、分配。': '，再直接勾选参数。',
        '已分配列超过 128：从图中移除不需要的列。': '已选列超过 128：切到对应图窗取消不需要的勾选，或清空该窗。',
        '或隐藏候选不会减少分配。': '会清空当前图窗，隐藏候选则保留曲线。',
        '长表分配或缩放后': '长表勾选或缩放后',
        '是否已分配': '是否已勾选参数',
        '完成分配后': '选好参数后',
        '清空选定图窗': '清空勾选',
        '每列只放在一个图窗。': '同一参数可在多个图窗中显示。',
        '分配': '选择',
    }
    # Apply residual terminology in untouched content without changing stable IDs.
    def visit(node):
        if node.get('type') == 'text':
            text = node['text']
            for before, after in replacements.items():
                text = text.replace(before, after)
            node['text'] = text
        for child in node.get('content', []):
            visit(child)
    visit(chapter['body'])
    # The association row spans several marked text nodes in the original table.
    for row in nodes['m02-table-files']['content']:
        if node_text(row['content'][0]) == '参数表关联':
            paragraph(row['content'][-1]['content'][0], '选择正确结果或未关联参数表。重新读表后，保留新表中存在的参数，图窗与右侧勾选保持一致。')
    for row in nodes['m02-table-figure']['content']:
        title = node_text(row['content'][0])
        if title == '曲线图例与悬停提示':
            paragraph(row['content'][1]['content'][0], '绘图区空白角落或图上方／列名与轴')
            paragraph(row['content'][-1]['content'][0], '结合颜色、线型和图例读取；双轴附 [左]/[右]。图例避让曲线，长名称完整换行。')
        elif title == '合并回首图':
            paragraph(row['content'][2]['content'][0], '仅后续图窗且多个图窗时出现')
    for block in blocks:
        if block['type'] == 'bulletList':
            for item in block.get('content', []):
                if node_text(item).startswith('右侧：'):
                    paragraph(item['content'][0], '右侧：选择当前图窗，搜索、筛选并勾选参数。勾选立即绘制，取消立即移除。')

    captures = [c for c in report['captures'] if c['anchor']]
    assert len(captures) == 16
    for capture in captures:
        raw = Path(capture['file']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == capture['sha256'] and capture['dpr'] == 1.5 and capture['pageZoom'] == 1
        asset_id = 'm02-current-' + capture['name'] + '-20261018-' + digest[:8]
        relative = 'assets/software-only/m02-current-20261018/' + asset_id + '.png'
        atomic(ROOT / 'manual' / relative, raw)
        asset = dict(id=asset_id, path=relative, kind='image', mime='image/png', sha256=digest,
                     width=capture['image'][0], height=capture['image'][1], caption=capture['caption'],
                     git=False, distribution='software-only', sourceType='实际应用导出' if capture['name'] == 'whole-figure' else '实际应用截图',
                     source='Windows 当前源码 Qt；合成4秒音频和参数，未使用研究材料。最大化、DPR1.5、页面100%，局部截图见capture。',
                     capture={'report': report_path.relative_to(ROOT).as_posix(), **{k: v for k, v in capture.items() if k != 'file'}})
        if not any(a['id'] == asset_id for a in project['assets']):
            project['assets'].append(asset)
        nodes[capture['anchor']]['attrs'].update(assetId=asset_id, caption=capture['caption'], alt=capture['caption'])
    # The controls overview is a second use of the current two-window capture.
    overview = nodes['m02-figure-m02-plots-flow-r3-light']['attrs']
    overview.update(assetId=nodes['m02-figure-second-plot-light']['attrs']['assetId'],
                    caption='点击图窗 2 后，右侧对应勾选 F2、F3。图窗 1 原有曲线保留，勾选操作直接作用于当前窗。',
                    alt='当前双图窗界面，图窗 2 高亮，右侧勾选与该窗曲线一致，分配及单独移除按钮已取消。')
    descriptor = next(c for c in project['chapters'] if c['id'] == 'm02')
    search, sections, target = [], [], None
    for block in blocks:
        if block['type'] == 'heading':
            target = block['attrs']['id']
            sections.append(dict(id=target, title=node_text(block), level=block['attrs'].get('level', 2)))
        text = node_text(block)
        if text:
            entry = dict(chapterId='m02', title=chapter['title'], text=text)
            if target:
                entry['targetId'] = target
            search.append(entry)
    descriptor['sections'] = sections
    index = project['searchIndex']
    first = next(i for i, entry in enumerate(index) if entry['chapterId'] == 'm02')
    project['searchIndex'] = index[:first] + search + [e for e in index[first:] if e['chapterId'] != 'm02']
    for path, value in [(chapter_path, chapter), (project_path, project)]:
        atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf8'))
    print(json.dumps(dict(figures=len(captures), assets=len(project['assets'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
