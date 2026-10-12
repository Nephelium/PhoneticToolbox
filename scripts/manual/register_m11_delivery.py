"""Register the reviewed native MFA installation capture, retaining old media."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from core import dump

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text('utf8'))
    capture = next(c for c in report['captures'] if c['name'] == 'component-install-light')
    assert report['success'] and capture['size'] == [2560, 1600]
    assert capture['dpr'] == 1.5 and capture['maximized'] and capture['pageZoom'] == 1 and capture['theme'] == 'light'
    source = report_path.parent / 'component-install-light.png'
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha == capture['sha256']
    aid = 'm11-component-delivery-r1-' + sha[:8]
    relative = 'assets/software-only/m11-delivery/' + aid + '.png'
    target = ROOT / 'manual' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest() == sha
    else:
        shutil.copyfile(source, target)
    pp, cp = ROOT / 'manual/project.json', ROOT / 'manual/chapters/m11.json'
    project, chapter = json.loads(pp.read_text('utf8')), json.loads(cp.read_text('utf8'))
    caption = '组件内已附普通话模型和词典。可从官网安装组件，也可选择下载好的 ZIP 或已解压文件夹；另有两个按钮打开官方资源目录，便于更换模型和词典。'
    asset = dict(id=aid, path=relative, kind='image', mime='image/png', sha256=sha,
                 width=2560, height=1600, caption=caption, git=False, distribution='software-only',
                 sourceType='实际应用截图', source='Windows 当前 Qt 源码，隔离组件目录，浅色、DPR 1.5、页面100%、最大化、全局侧栏收起。',
                 capture={'report': report_path.relative_to(ROOT).as_posix(), **capture})
    assets = {a['id']: a for a in project['assets']}
    assets[aid] = asset
    project['assets'] = list(assets.values())
    node = next(n for n in chapter['body']['content'] if n.get('attrs', {}).get('id') == 'm11-figure-component-manager')
    node['attrs'].update(assetId=aid, caption=caption, alt='MFA 组件安装三个入口和官方声学模型、发音词典下载按钮。')
    chapter['status'] = 'reviewed'
    next(c for c in project['chapters'] if c['id'] == 'm11')['status'] = 'reviewed'
    cp.write_bytes(dump(chapter))
    pp.write_bytes(dump(project))
    print(json.dumps({'asset': aid, 'sha256': sha}))


if __name__ == '__main__':
    main()
