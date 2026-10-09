"""Register the reviewed M14 full-window capture report without deleting history."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

from core import dump, node_text, safe_file

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / 'manual'


def walk(node):
    yield node
    for child in node.get('content', []):
        yield from walk(child)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-report', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.capture_report.read_text(encoding='utf-8'))
    assert report['success'] is True and len(report['captures']) == 17
    project = json.loads((MANUAL / 'project.json').read_text(encoding='utf-8'))
    chapter = json.loads((MANUAL / 'chapters/m14.json').read_text(encoding='utf-8'))
    figures = {n['attrs']['id']: n for n in walk(chapter['body']) if n['type'] == 'image'}
    assert set(figures) == {c['figureId'] for c in report['captures']}
    assets = {a['id']: a for a in project['assets']}
    for capture in report['captures']:
        assert capture['window'] == capture['image'] == capture['viewport'] == [2560, 1440]
        assert capture['captureMode'] == 'user-authorized-2560x1440-window' and capture['theme'] == 'light'
        source = Path(capture['file'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == capture['sha256']
        relative = 'assets/software-only/screenshots/' + capture['id'] + '.png'
        target = safe_file(MANUAL, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == capture['sha256']
        else:
            shutil.copyfile(source, target)
        node = figures[capture['figureId']]
        node['attrs']['assetId'] = capture['id']
        assets[capture['id']] = dict(id=capture['id'], path=relative, kind='image', mime='image/png',
                                     sha256=capture['sha256'], caption=node['attrs']['caption'],
                                     alt=node['attrs']['alt'], width=2560, height=1440,
                                     git=False, distribution='software-only', sourceType='实际应用完整窗口截图',
                                     source='Windows Qt 完整应用窗口，采用作者指定的 2560×1440 尺寸；源表为测试.xlsx，调类名称暂拟，仅演示操作。')
    project['assets'] = list(assets.values())
    (MANUAL / 'chapters/m14.json').write_bytes(dump(chapter))
    # Limit index changes to the edited chapter and keep unrelated entries intact.
    descriptor = next(d for d in project['chapters'] if d['id'] == 'm14')
    descriptor['sections'] = [dict(id=n['attrs']['id'], title=node_text(n), level=n['attrs']['level'])
                              for n in walk(chapter['body']) if n['type'] == 'heading']
    replacement = []
    for node in chapter['body']['content']:
        text = node_text(node)
        if text:
            replacement.append(dict(chapterId='m14', title=descriptor['title'], text=text,
                                    **({'targetId': node['attrs']['id']} if node.get('attrs', {}).get('id') else {})))
    old_index = project.get('searchIndex', [])
    first = next((i for i, entry in enumerate(old_index) if entry['chapterId'] == 'm14'), len(old_index))
    kept = [entry for entry in old_index if entry['chapterId'] != 'm14']
    project['searchIndex'] = kept[:first] + replacement + kept[first:]
    (MANUAL / 'project.json').write_bytes(dump(project))

    # A standalone chapter preview for direct local review, using the registered
    # full-size originals. It is supplementary to the application's own reader.
    preview = Path(report['out']) / '音系归纳说明书预览.html'
    figure_count = 0

    def render(node):
        nonlocal figure_count
        kind = node['type']
        attrs = node.get('attrs', {})
        if kind == 'text':
            text = html.escape(node.get('text', ''))
            for mark in node.get('marks', []):
                tag = {'bold': 'strong', 'italic': 'em', 'code': 'code', 'subscript': 'sub', 'superscript': 'sup'}.get(mark['type'])
                if tag:
                    text = f'<{tag}>{text}</{tag}>'
            return text
        if kind == 'image':
            figure_count += 1
            capture = next(c for c in report['captures'] if c['id'] == attrs['assetId'])
            src = 'screenshots/' + Path(capture['file']).name
            return '<figure id="' + html.escape(attrs['id']) + '"><a href="' + src + '"><img loading="lazy" src="' + src + '" alt="' + html.escape(attrs.get('alt', '')) + '"></a><figcaption>图 ' + str(figure_count) + '　' + html.escape(attrs['caption']) + '</figcaption></figure>'
        children = ''.join(render(child) for child in node.get('content', []))
        if kind == 'heading':
            level = min(6, max(2, attrs.get('level', 2)))
            return f'<h{level} id="{html.escape(attrs["id"])}">{children}</h{level}>'
        if kind == 'codeBlock':
            return '<pre><code>' + children + '</code></pre>'
        tags = {'doc': 'main', 'paragraph': 'p', 'orderedList': 'ol', 'bulletList': 'ul',
                'listItem': 'li', 'table': 'table', 'tableRow': 'tr', 'tableCell': 'td', 'tableHeader': 'th', 'blockquote': 'blockquote'}
        if kind in tags:
            tag = tags[kind]
            span = ' colspan="' + str(attrs['colspan']) + '"' if attrs.get('colspan', 1) > 1 else ''
            return f'<{tag}{span}>{children}</{tag}>'
        if kind == 'hardBreak':
            return '<br>'
        if kind == 'horizontalRule':
            return '<hr>'
        if kind in ('citation', 'crossReference'):
            return html.escape(str(attrs.get('label', attrs.get('targetId', attrs.get('referenceId', '')))))
        raise ValueError('Unsupported preview node: ' + kind)

    body = render(chapter['body'])
    preview.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>音系归纳使用说明</title><style>body{margin:0;background:#f7f8fa;color:#202a35;font:16px/1.85 "Microsoft YaHei",sans-serif}header,main{max-width:1500px;margin:auto;padding:24px 42px;background:#fff}header{border-bottom:1px solid #d9e0e7}h1{margin:0}h2{margin-top:40px;border-bottom:1px solid #d9e0e7;padding-bottom:8px}h3,h4{margin-top:28px}figure{margin:28px 0}img{display:block;width:100%;height:auto;border:1px solid #d9e0e7}figcaption{font-size:14px;color:#536273;margin-top:8px}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}th,td{border:1px solid #ccd5df;padding:8px 12px;vertical-align:top}th{background:#edf3fa}pre{overflow:auto;background:#f2f5f8;padding:18px;line-height:1.7}td p{margin:0}li{margin:8px 0}a{color:#1663af}code{font-family:Consolas,monospace}@media(max-width:700px){header,main{padding:20px}}</style><header><h1>音系归纳使用说明</h1><p>本章示例采用测试.xlsx。暂拟调类仅用于展示操作，不代表真实调类。全部17张截图采用2560×1440完整应用窗口，点击图片可查看原图。</p></header>' + body + '</html>', encoding='utf-8')
    print(json.dumps(dict(registered=17, preview=str(preview), chapter='m14'), ensure_ascii=False))


if __name__ == '__main__':
    main()
