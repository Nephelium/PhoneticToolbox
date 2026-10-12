"""Synchronize four settings figures and the affected user instructions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def paragraph(text):
    return dict(type='paragraph', content=[dict(type='text', text=text)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    report_path = parser.parse_args().report.resolve(strict=True)
    report = json.loads(report_path.read_text('utf-8'))
    assert report['success']
    project_path = ROOT / 'manual/project.json'
    project = json.loads(project_path.read_text('utf-8'))
    chapter_path = ROOT / 'manual/chapters/settings.json'
    chapter = json.loads(chapter_path.read_text('utf-8'))
    nodes = chapter['body']['content']
    chapter['title'] = '设置、主题、字体与图标'
    descriptor = next(c for c in project['chapters'] if c['id'] == 'settings')
    descriptor['title'] = chapter['title']
    nodes[1] = paragraph('在工作台左下角点击“设置”，设置会作为独立标签打开。顶部有“外观”“字体”“图标”“本机存储与自动清理”四个按钮，每次显示所选页面。点击切换，也可聚焦分类按钮后用左右方向键切换，Home、End 跳到首尾。页面足够宽时组内分为左右两栏，窄窗口自动回到单栏。外观与图标选择立即生效并保存在本机；字体需单独点击应用。切换分类或工作台标签会保留本次字体编辑和清理期限输入。')
    def revise(value):
        if isinstance(value, dict):
            if value.get('type') == 'text':
                value['text'] = value.get('text', '').replace('初始方案是 Codex。', '初始方案是 Everforest。已经保存的其他配色会继续沿用。').replace('左右两张浅色、深色示意卡', '跟随系统、浅色、深色三张示意卡').replace('默认是黄色底的 K1-A 波形。', '默认是黄色底、雾紫声波与深蓝末端的 W1。').replace('回到 K1-A。', '回到黄底、雾紫声波与深蓝末端的 W1。')
            for child in value.values():
                revise(child)
        elif isinstance(value, list):
            for child in value:
                revise(child)
    revise(chapter)
    if not any(n.get('attrs', {}).get('id') == 'settings-icons' for n in nodes):
        insert = next(i for i,n in enumerate(nodes) if n.get('attrs', {}).get('id') == 'settings-retention')
        nodes[insert:insert] = [
            dict(type='heading', attrs=dict(id='settings-icons', level=2), content=[dict(type='text', text='应用图标')]),
            paragraph('打开顶部“图标”页，在连续网格中点击喜欢的图片。内置共 31 款，同类相邻排列：波形 7 款、语谱 1 款、口形 1 款、标注 2 款、IPA 20 款。格子只显示图片，选中后边框高亮。默认是黄色底、雾紫声波与深蓝末端的 W1。右侧可查看浅色、深色背景与小尺寸效果。'),
            paragraph('点击“导入图片”，选择 PNG、JPG、WebP 或 ICO 文件，单张不超过 10 MB、1600 万像素。图片等比例居中到图标区域，保留透明部分，导入后立即选中。最多保留 8 张自定义图片，相同图片重复导入会复用已有项。无法读取或保存时提示原因并保留原选择。'),
            paragraph('“恢复默认图标”回到黄底、雾紫声波与深蓝末端的 W1。“移除”只移除保存的自定义图标，原始图片保留；移除正在使用的图标时回到默认项。桌面版的应用内图标和窗口图标同步改变，选择在重启后保留；网页预览保存在当前浏览器。Windows 已固定的任务栏入口、桌面快捷方式和已有 EXE 的内嵌图标各自独立，已有入口不随本页选择自动改写。'),
            dict(type='image', attrs=dict(id='settings-icons-figure', assetId='', caption='', width='100%', align='center', alt='')),
        ]
    captions = {
        'appearance': ('settings-figure-settings-r2-light', '外观页默认使用 Everforest。跟随系统、浅色、深色三张预览与显示模式按钮上下对齐；宽窗口中，配色与显示模式在左栏，缩放、按钮与波形颜色在右栏。'),
        'fonts': ('settings-figure-settings-v3-dark', '字体页左栏编辑字体与字号，右栏预览。切换设置分类保留编辑，应用按钮统一保存。'),
        'icons': ('settings-icons-figure', '图标页用一个紧凑网格排列 31 款内置图标，同类连续相邻，只显示图片。默认是黄底、雾紫声波与深蓝末端的 W1，右栏提供预览与恢复默认。'),
        'storage': (None, '本机存储与自动清理页在宽窗口分为两栏，左栏编辑保留政策，右栏显示占用与操作。'),
    }
    for name, (figure_id, caption) in captions.items():
        capture = next(c for c in report['captures'] if c['name'] == name)
        assert capture['image'] == [2560,1600] and capture['dpr'] == 1.5 and capture['maximized'] and capture['pageZoom'] == 1 and capture['theme'] == 'light'
        source = Path(capture['file'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == capture['sha256']
        identity = 'settings-icons-' + name + '-' + digest[:8]
        relative = 'assets/software-only/settings-icons/' + identity + '.png'
        target = ROOT / 'manual' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source,target)
        if not any(a['id'] == identity for a in project['assets']):
            project['assets'].append(dict(id=identity, path=relative, kind='image', mime='image/png', sha256=digest, width=2560, height=1600, caption=caption, git=False, distribution='software-only', sourceType='实际应用截图', source='Windows 当前源码 Qt；默认 Everforest/W1雾紫声波与深蓝末端，浅色、最大化、原生 DPR1.5、页面100%，隔离测试偏好与现有schema库副本。', capture={'report':report_path.relative_to(ROOT).as_posix(), **{k:v for k,v in capture.items() if k!='file'}}))
        figure = next(n for n in nodes if n['type']=='image' and (n['attrs'].get('id')==figure_id if figure_id else 'storage' in n['attrs']['assetId']))
        figure['attrs'].update(assetId=identity,caption=caption,alt=caption)
    getting_path = ROOT / 'manual/chapters/getting-started.json'
    getting = json.loads(getting_path.read_text('utf-8'))
    def update_getting(value):
        if isinstance(value,dict):
            if value.get('type')=='text':
                text = value.get('text','').replace('设置页顶部还有“字体”和“本机存储与自动清理”两个分类按钮。','设置页顶部还有“字体”“图标”“本机存储与自动清理”三个分类按钮，宽窗口分左右栏。默认配色是 Everforest，已有配色选择保留。').replace('默认使用 K1-A 波形。', '默认使用黄底、雾紫声波与深蓝末端的 W1 波形。')
                if '在“图标”页选择内置图标' not in text:
                    text = text.replace('在“配色方案”中选配色','在“图标”页选择内置图标或导入自己的图片，默认使用黄底、雾紫声波与深蓝末端的 W1 波形。在“配色方案”中选配色')
                value['text'] = text
            for child in value.values():update_getting(child)
        elif isinstance(value,list):
            for child in value:update_getting(child)
    update_getting(getting)
    # Refresh the introductory home/reader figures from the same native capture.
    for name in ('home', 'manual'):
        capture = next(c for c in report['captures'] if c['name'] == name)
        source = Path(capture['file'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == capture['sha256']
        identity = 'settings-brand-' + name + '-' + digest[:8]
        relative = 'assets/software-only/settings-icons/' + identity + '.png'
        target = ROOT / 'manual' / relative
        if not target.exists():
            shutil.copyfile(source, target)
        if not any(a['id'] == identity for a in project['assets']):
            project['assets'].append(dict(id=identity, path=relative, kind='image', mime='image/png', sha256=digest, width=2560, height=1600, caption='当前应用首页' if name == 'home' else '当前应用说明书', git=False, distribution='software-only', sourceType='实际应用截图', source='当前源码 Qt，默认雾紫声波与深蓝末端图标；隔离测试偏好。', capture={'report':report_path.relative_to(ROOT).as_posix(), **{k:v for k,v in capture.items() if k!='file'}}))
        for node in getting['body']['content']:
            if node.get('type') == 'image' and ('shell-' + name + '-' in node['attrs'].get('assetId', '') or 'settings-brand-' + name + '-' in node['attrs'].get('assetId', '')):
                node['attrs']['assetId'] = identity
    for path,data in [(project_path,project),(chapter_path,chapter),(getting_path,getting)]:
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Registered four native figures and updated settings/getting-started instructions')


if __name__ == '__main__':
    main()
