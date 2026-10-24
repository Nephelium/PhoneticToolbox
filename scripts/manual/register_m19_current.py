"""Register reviewed M19 screenshots and update only this chapter's media/index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from core import dump, node_text
from build import write_atomic

ROOT = Path(__file__).resolve().parents[2]
FIGURES = {
    'data-download': ('m19-start', '首次打开语言地图，点击下载资料取得地图资料。已有完整资料时可选择资料夹；完成载入后，资料保存在本机，重新打开可离线使用。'),
    'dialect-distribution': ('m19-classification', '在省级地区选择广东后，点击潮安区的整理记录，地图定位到当地，右侧显示完整方言分类和原始行号。'),
    'phonology-broad-filter': ('m19-groups', '阳声韵尾的演变按大类显示。点击图例中的有[-m]，地图和表格只保留这一类地点，再点全部类别恢复。'),
    'two-maps': ('m19-maps', '双图比较调类的数目与阳声韵尾的演变。两幅图共享视野，第二幅图从标题栏选择项目。'),
    'four-maps': ('m19-maps', '四图同时比较调类的数目、撮口呼韵母、阳声韵尾的演变与太阳的说法。每幅图可独立选择项目，拖动和缩放时保持同一视野。'),
    'two-place-comparison': ('m19-similarity', '北京与广州的地点对照。项目范围选择语音，记录关系选择记录不同；导出对照表会保存当前筛选范围的全部记录。'),
    'custom-point-editor': ('m19-edit-export', '输入地点名称、WGS84经纬度、类别或IPA与备注，再保存地点。图中珠海示例点及aː˧仅演示填写方式，研究时请替换为实测资料。'),
    'custom-map-export': ('m19-edit-export', '自建的珠海、广州示例点保存在当前工程中。可导出工程继续编辑，或用SVG、PNG和导出表格保存研究成果；两点IPA为填写演示。'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text('utf-8'))
    assert report['success'] and {c['name'] for c in report['captures']} == set(FIGURES)
    manifest_path = Path(report['dataSource']) / 'data-manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == report['dataManifestSha256']
    manifest = json.loads(manifest_path.read_text('utf-8'))
    datav = manifest.get('boundaries', {}).get('version', '').startswith('datav-')
    source_ids = ['M19-ZENODO', 'M19-SHAOZI', 'M19-DATAV', 'M19-NLSC', 'M19-NATURAL-EARTH'] if datav else ['M19-ZENODO', 'M19-SHAOZI', 'M19-OSM', 'M19-NLSC', 'M19-HK-HAD', 'M19-NATURAL-EARTH']
    rights_basis = ('当前应用界面与演示编辑由本项目生成；两份语言数据按CC BY 4.0使用。底图分别保留DataV GeoAtlas／高德开放平台的学习交流用途告知、国土测绘中心的政府资料开放授权条款第1版，以及Natural Earth公共领域署名。截图未使用原书扫描、自然录音及受限旧媒体。' if datav else '当前应用界面与演示编辑由本项目生成；两份语言数据按CC BY 4.0使用。底图保留OpenStreetMap、国土测绘中心、香港政府及Natural Earth署名及各自许可。原书扫描、自然录音及受限旧媒体均未使用。')
    project_path, chapter_path = ROOT / 'manual/project.json', ROOT / 'manual/chapters/m19.json'
    project_before, chapter_before = project_path.read_bytes(), chapter_path.read_bytes()
    project = json.loads(project_before)
    chapter = json.loads(chapter_before)
    content = chapter['body']['content']
    assets = {a['id']: a for a in project['assets']}
    by_name = {c['name']: c for c in report['captures']}
    for name, (anchor, caption) in FIGURES.items():
        capture = by_name[name]
        source = report_path.parent / capture['file']
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == capture['sha256']
        assert capture['image'] == [2560, 1600] and capture['dpr'] == 1.5
        assert capture['pageZoom'] == 1 and capture['maximized']
        assert capture['theme'] == 'light' and capture['navigationCollapsed']
        asset_id = 'm19-current-' + name + '-' + digest[:8]
        relative = 'assets/public/m19-current-20261024/' + asset_id + '.png'
        target = ROOT / 'manual' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        else:
            shutil.copyfile(source, target)
        metadata = {k: v for k, v in capture.items() if k not in ('file', 'visibleText')}
        assets[asset_id] = dict(id=asset_id, path=relative, kind='image', mime='image/png',
            sha256=digest, width=2560, height=1600, caption=caption, git=True,
            distribution='public', sourceType='实际应用截图',
            source='Windows当前源码Qt，浅色、原生DPR1.5、页面100%、最大化、收起导航；中性语音与太阳词汇项目及明确标示的自建示例点，无原书扫描页。',
            rightsBasis=rights_basis, sourceIds=source_ids,
            capture=dict(report=report_path.relative_to(ROOT).as_posix(), **metadata))
        figure_id = 'm19-current-' + name + '-figure'
        node = next((n for n in content if n.get('attrs', {}).get('id') == figure_id), None)
        if node is None:
            node = dict(type='image', attrs=dict(id=figure_id))
            start = next(i for i, n in enumerate(content) if n.get('attrs', {}).get('id') == anchor)
            end = next((i for i in range(start + 1, len(content)) if content[i]['type'] == 'heading'), len(content))
            content.insert(end, node)
        node['attrs'].update(assetId=asset_id, caption=caption, alt=caption, width='100%', align='center')

    def text(value, href=None):
        node = dict(type='text', text=value)
        if href:
            node['marks'] = [dict(type='link', attrs=dict(href=href))]
        return node

    attribution = dict(type='paragraph', attrs=dict(id='m19-screenshot-attribution'), content=[
        text('本章操作图由当前应用生成，语言数据来自 '),
        text('Huang、Grieve 与 Jiao（2024）', 'https://zenodo.org/records/10697975'),
        text('及'), text('井韶子整理数据', 'https://zenodo.org/records/15897647'),
        text('，两份数据均按 '), text('CC BY 4.0', 'https://creativecommons.org/licenses/by/4.0/'),
        text('使用。图中显示方式、筛选结果和自建演示地点由本应用生成。底图：'),
        text('© OpenStreetMap contributors', 'https://www.openstreetmap.org/copyright'),
        text('（2026-10-22，'), text('ODbL 1.0', 'https://opendatacommons.org/licenses/odbl/1-0/'),
        text('）；内政部国土测绘中心（COUNTY_MOI_1090820、TOWN_MOI_1120317，'),
        text('政府资料开放授权条款第1版', 'https://data.gov.tw/license'),
        text('）；© 香港特别行政区政府／民政事务总署及 DATA.GOV.HK（18区边界，2026-09-11，'),
        text('使用条款1.2', 'https://data.gov.hk/en/terms-and-conditions'),
        text('）；'), text('Natural Earth', 'https://www.naturalearthdata.com/about/terms-of-use/'),
        text('（公共领域）。行政几何经过坐标转换、显示简化及明确的聚合处理，各来源保留各自许可。'),
    ])
    if datav:
        attribution['content'] = [
            text('本章操作图由当前应用生成，语言数据来自 '),
            text('Huang、Grieve 与 Jiao（2024）', 'https://zenodo.org/records/10697975'),
            text('及'), text('井韶子整理数据', 'https://zenodo.org/records/15897647'),
            text('，两份数据均按 '), text('CC BY 4.0', 'https://creativecommons.org/licenses/by/4.0/'),
            text('使用。图中显示方式、筛选结果和自建演示地点由本应用生成。底图：'),
            text('DataV GeoAtlas／高德开放平台', 'https://help.aliyun.com/zh/datav/datav-7-0/user-guide/datav-geoatlas-widgets/'),
            text('（areas_v3，页面标注2021.5，仅供学习和交流）；'),
            text('内政部国土测绘中心', 'https://data.gov.tw/dataset/7441'),
            text('（TOWN_MOI_1120317，'), text('政府资料开放授权条款第1版', 'https://data.gov.tw/license'),
            text('）；'), text('Natural Earth', 'https://www.naturalearthdata.com/about/terms-of-use/'),
            text('（公共领域）。各来源分别适用其用途告知或许可。'),
        ]
    existing = next((i for i, n in enumerate(content) if n.get('attrs', {}).get('id') == 'm19-screenshot-attribution'), None)
    if existing is None:
        at = next(i for i, n in enumerate(content) if n.get('attrs', {}).get('id') == 'm19-sources')
        content.insert(at + 1, attribution)
    else:
        content[existing] = attribution

    project['assets'] = list(assets.values())
    chapter['status'] = 'reviewed'
    descriptor = next(c for c in project['chapters'] if c['id'] == 'm19')
    descriptor['status'] = chapter['status']
    descriptor['sections'] = [dict(id=n['attrs']['id'], title=node_text(n), level=n['attrs']['level'])
        for n in content if n['type'] == 'heading' and n['attrs'].get('id')]
    # Refresh this chapter alone, preserving other authors' search entries byte-for-byte in value.
    target_id = None
    search = []
    for node in content:
        if node['type'] == 'heading':
            target_id = node.get('attrs', {}).get('id')
        text = node_text(node)
        if text:
            entry = dict(chapterId='m19', title=chapter['title'], text=text)
            if target_id:
                entry['targetId'] = target_id
            search.append(entry)
    previous = project.get('searchIndex', [])
    insert = next((i for i, e in enumerate(previous) if e['chapterId'] == 'm19'), len(previous))
    preserved = [e for e in previous if e['chapterId'] != 'm19']
    project['searchIndex'] = preserved[:insert] + search + preserved[insert:]
    if project_path.read_bytes() != project_before or chapter_path.read_bytes() != chapter_before:
        raise RuntimeError('Manual source changed during registration; retry with the current project')
    write_atomic(chapter_path, dump(chapter))
    write_atomic(project_path, dump(project))
    print(json.dumps(dict(chapter='m19', registered=len(FIGURES), images=sum(n['type'] == 'image' for n in content))))


if __name__ == '__main__':
    main()
