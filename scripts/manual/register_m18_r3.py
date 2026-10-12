"""Register current verified M18 screenshots without deleting prior formal media."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from core import dump
ROOT=Path(__file__).resolve().parents[2]
CAPTIONS={
 'reader-collapsed':'服务器推荐采用单行目录，与个人 PDF 分开；缩放菜单保留适宽，顶部提供自动获取设置。',
 'reader-details':'展开论文信息，查看双语题名、作者、固定版本、许可、翻译说明和阅读导引。',
 'reader-fullscreen':'全屏阅读保留原文、中文译文、论文解读及阅读工具，点击退出全屏返回原窗口。',
 'reader-annotations':'选中文字添加荧光笔，右侧批注列表可定位标记；批注按文档内容保存在本机。',
 'reader-export':'导出窗口选择文档和是否包含本机批注，随后由原生窗口选择保存位置。',
 'reader-citations':'选择引用格式，核对作者、固定版本与来源日期后复制。',
 'reader-archive':'往期左侧选择年份、月份，右侧勾选双语题名，只获取已选论文；已获取项默认隐藏。',
 'reader-analysis':'独立论文解读使用同一阅读器，分析与原论文结果分别标明。',
 'reader-schedule':'自动获取开启时应用启动检查一次，可修改持续运行期间的检查间隔。此例保存为 12 小时。',
 'reader-local':'在我的 PDF 中导入麦耘论文及 extIPA 文件副本，目录与服务器推荐独立。',
 'reader-local-folder':'收藏夹对应本地真实子文件夹，移至方言元音后可按收藏夹筛选，原导入来源保留。',
}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);args=parser.parse_args();args.report=args.report.resolve()
 report=json.loads(args.report.read_text('utf8'));assert report['success'] and report['sourceHashesUnchanged']
 project_path=ROOT/'manual/project.json';chapter_path=ROOT/'manual/chapters/m18.json'
 project=json.loads(project_path.read_text('utf8'));chapter=json.loads(chapter_path.read_text('utf8'))
 assets={a['id']:a for a in project['assets']};ids={}
 for capture in report['captures']:
  name=capture['name']
  if name not in CAPTIONS:continue
  assert capture['maximized'] or name=='reader-fullscreen'
  source=args.report.parent/(name+'.png');raw=source.read_bytes();sha=hashlib.sha256(raw).hexdigest()
  aid='m18-r3-'+name+'-'+sha[:8];ids[name]=aid;relative='assets/software-only/m18-r3/'+aid+'.png'
  target=ROOT/'manual'/relative;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
  else:shutil.copyfile(source,target)
  assets[aid]=dict(id=aid,path=relative,kind='image',mime='image/png',sha256=sha,width=capture['size'][0],height=capture['size'][1],caption=CAPTIONS[name],git=False,distribution='software-only',sourceType='实际应用截图',source='当前 Windows Qt 源码，深色主题，原生 DPR 1.5，页面 100%，无缩放或裁剪。用户指定的个人 PDF 仅用于本地授权演示副本，不提供原文再分发。',capture={'report':args.report.relative_to(ROOT).as_posix(),**capture})
 assert set(ids)==set(CAPTIONS)
 content=chapter['body']['content'];template=next(n for n in content if n['type']=='image')
 for node in content:
  if node['type']=='image':
   old=node['attrs']['assetId']
   name=next((n for n in ids if old.startswith('m18-r2-'+n+'-')),None)
   if name:node['attrs'].update(assetId=ids[name],caption=CAPTIONS[name],alt=CAPTIONS[name])
 after={'m18-new-papers':'reader-schedule','m18-history':'reader-archive','m18-local-import':'reader-local','m18-local-folders':'reader-local-folder','m18-guide':'reader-analysis'}
 for section,name in after.items():
  start=next(i for i,n in enumerate(content) if n['type']=='heading' and n['attrs']['id']==section)
  end=next((i for i in range(start+1,len(content)) if content[i]['type']=='heading'),len(content))
  if any(n['type']=='image' and n['attrs'].get('assetId')==ids[name] for n in content[start:end]):continue
  node=json.loads(json.dumps(template));node['attrs'].update(assetId=ids[name],id=ids[name]+'-figure',caption=CAPTIONS[name],alt=CAPTIONS[name]);content.insert(end,node)
 seen=set()
 for node in content:
  if node['type']=='image':
   identity=node['attrs'].get('id')
   if identity in seen:node['attrs']['id']=node['attrs']['assetId']+'-figure'
   seen.add(node['attrs'].get('id'))
 chapter['status']='reviewed';project['assets']=list(assets.values())
 next(c for c in project['chapters'] if c['id']=='m18')['status']='reviewed'
 chapter_path.write_bytes(dump(chapter));project_path.write_bytes(dump(project));print(json.dumps({'registered':ids},ensure_ascii=False))
if __name__=='__main__':main()
