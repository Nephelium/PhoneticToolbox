"""Register current compact-shell figures and revise affected reader instructions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    report_path=parser.parse_args().report.resolve(strict=True)
    report=json.loads(report_path.read_text('utf-8'));assert report['success']
    project_path=ROOT/'manual/project.json';project=json.loads(project_path.read_text('utf-8'))
    paths={key:ROOT/'manual/chapters'/f'{key}.json' for key in ('getting-started','settings','m18')}
    chapters={key:json.loads(path.read_text('utf-8')) for key,path in paths.items()}
    replacements={
        'Windows 11 的窗口标题栏会与菜单、标签顶栏同步配色。':'Windows 顶边将图标、名称、标签与窗口控制放在同一行，随菜单配色变化。标签文字与关闭按钮同色。拖动顶行空白或品牌区可移动窗口，双击可最大化或还原；窗口边缘可调整大小。开源与学术致谢位于首页内容区右上角。',
        '点击首页卡片或左侧导航里的工具。顶部出现同名标签；回到“首页”不会关闭工具。':'点击首页卡片或左侧导航里的工具。顶部出现同名标签；回到“首页”不会关闭工具。Windows 的品牌与标签放在同一顶行，开源与学术致谢入口位于首页内容区右上角。',
        '在说明书左侧点章节或小节，正文滚动到目标位置并短暂高亮。也可用“搜索说明书”输入控件名或操作词；窄窗先点“目录与搜索”。':'在说明书左侧点章节或小节，正文滚动到目标位置，标题文字短暂变色；目录当前项以底色和文字颜色提示。拖动目录与正文之间的边界可调整宽度，切换标签后保留。也可用“搜索说明书”输入控件名或操作词；窄窗先点“目录与搜索”。',
        '工具导航与主区之间的分界线也可拖动。':'工具导航与主区之间的分界线也可拖动，使用说明目录和语音学论文精读的论文库同样支持，各自保存宽度。',
        '两个来源的目录与收藏夹独立管理，共用连续阅读、选字、批注和导出工具。切换来源后，题名搜索、年月与收藏夹筛选回到全部范围。':'两个来源的目录与收藏夹独立管理，共用连续阅读、选字、批注和导出工具。切换来源后，题名搜索、年月与收藏夹筛选回到全部范围。拖动论文库与阅读区之间的边界可调整左栏宽度，放开后自动保存；切换来源或工作台标签继续沿用，窗口临时变窄不会覆盖保存的宽度。',
    }
    def revise(value):
        if isinstance(value,dict):
            if value.get('type')=='text':
                text=value.get('text','')
                text=text.replace('开源与学术致谢入口位于右上角。','开源与学术致谢入口位于首页内容区右上角。').replace('开源与学术致谢位于右上角。','开源与学术致谢位于首页内容区右上角。')
                for old,new in replacements.items():
                    if new not in text:text=text.replace(old,new)
                value['text']=text
            for child in value.values():revise(child)
        elif isinstance(value,list):
            for child in value:revise(child)
    for chapter in chapters.values():revise(chapter)
    figures=[n for n in chapters['getting-started']['body']['content'] if n['type']=='image']
    for name,index,caption in [('home',0,'Windows 顶边集中了品牌、标签与窗口按钮，开源与学术致谢在首页内容区右上角。首页与导航按工具分类提供入口。'),('manual',1,'使用说明的目录选中项以颜色提示，正文定位标题不显示框线。目录与正文之间的边界可以左右拖动。')]:
        capture=next(c for c in report['captures'] if c['name']==name)
        assert capture['image']==[2560,1600] and capture['dpr']==1.5 and capture['pageZoom']==1 and capture['maximized']
        source=Path(capture['file']);digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest==capture['sha256']
        identity='compact-shell-'+name+'-'+digest[:8]
        relative='assets/software-only/compact-shell/'+identity+'.png';target=ROOT/'manual'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(source,target)
        if not any(a['id']==identity for a in project['assets']):
            project['assets'].append(dict(id=identity,path=relative,kind='image',mime='image/png',sha256=digest,width=2560,height=1600,caption=caption,git=False,distribution='software-only',sourceType='实际应用截图',source='Windows源码Qt，Everforest浅色，原生DPR1.5，页面100%，隔离测试偏好和既有schema库副本。',capture={'report':report_path.relative_to(ROOT).as_posix(),**{k:v for k,v in capture.items() if k!='file'}}))
        figures[index]['attrs'].update(assetId=identity,caption=caption,alt=caption)
    for key,path in paths.items():path.write_text(json.dumps(chapters[key],ensure_ascii=False,indent=2)+'\n','utf-8')
    temporary=project_path.with_suffix('.compact-shell.tmp')
    temporary.write_text(json.dumps(project,ensure_ascii=False,indent=2)+'\n','utf-8')
    temporary.replace(project_path)
    print('Registered compact home/manual figures and revised three chapters')


if __name__=='__main__':main()
