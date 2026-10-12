"""Register reviewed native paper screenshots; preserve existing formal media."""
import argparse,hashlib,json,shutil
from pathlib import Path
from core import dump
ROOT=Path(__file__).resolve().parents[2]
CAPTURES={
 'reader-new-details':('m18-guide','第二篇推荐显示原稿2024年日期、固定版本和CC BY 4.0许可，栏目2026年新收录与论文原始发表时间分别说明。'),
 'reader-content-check':('m18-content-updates','检查完成后显示服务器目录篇数。本例两篇已经完整获取，因此无需再次下载。'),
 'reader-remove-recommendation':('m18-remove-folder','删除服务器推荐收藏夹前核对说明，论文回到未收藏，已获取文件与批注保留。'),
 'reader-remove-local':('m18-remove-folder','删除本地收藏夹时，内部所有文件和子文件夹移到上一级，不删除PDF。'),
 'reader-remove-collision':('m18-remove-folder','上一级存在同名项目时整批停止，文件未移动、收藏夹未删除，处理重名后可重试。'),
 'reader-remove-local-result':('m18-remove-folder','删除完成后原PDF继续打开，文件与内部子文件夹保留在上一级。'),
}
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);args=p.parse_args();report_path=args.report.resolve();r=json.loads(report_path.read_text('utf8'));assert r['success']
 pp=ROOT/'manual/project.json';cp=ROOT/'manual/chapters/m18.json';project=json.loads(pp.read_text('utf8'));chapter=json.loads(cp.read_text('utf8'));assets={a['id']:a for a in project['assets']};content=chapter['body']['content'];template=next(n for n in content if n['type']=='image')
 for name,(section,caption) in CAPTURES.items():
  capture=next(c for c in r['captures'] if c['name']==name);assert capture['size']==[2560,1600] and capture['dpr']==1.5 and capture['pageZoom']==1 and capture['maximized'] and capture['theme']=='light'
  source=report_path.parent/(name+'.png');sha=hashlib.sha256(source.read_bytes()).hexdigest();aid='m18-r4-'+name+'-'+sha[:8];relative='assets/software-only/m18-r4/'+aid+'.png';target=ROOT/'manual'/relative;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
  else:shutil.copyfile(source,target)
  assets[aid]=dict(id=aid,path=relative,kind='image',mime='image/png',sha256=sha,width=2560,height=1600,caption=caption,git=False,distribution='software-only',sourceType='实际应用截图',source='Windows当前Qt源码，隔离合成目录及CC BY 4.0公开论文副本，原生DPR 1.5，浅色主题，页面100%，最大化。Qt原图2561×1601时仅裁右下各一像素。',capture={'report':report_path.relative_to(ROOT).as_posix(),**capture})
  start=next(i for i,n in enumerate(content) if n['type']=='heading' and n['attrs']['id']==section);end=next((i for i in range(start+1,len(content)) if content[i]['type']=='heading'),len(content))
  if not any(n['type']=='image' and n['attrs'].get('assetId')==aid for n in content[start:end]):
   node=json.loads(json.dumps(template));node['attrs'].update(assetId=aid,id=aid+'-figure',caption=caption,alt=caption);content.insert(end,node)
 project['assets']=list(assets.values());chapter['status']='reviewed'
 entry=next(c for c in project['chapters'] if c['id']=='m18');entry['status']='reviewed'
 entry['sections']=[dict(id=n['attrs']['id'],title=''.join(t.get('text','') for t in n.get('content',[])),level=n['attrs']['level']) for n in content if n['type']=='heading' and n['attrs']['level']>=2]
 cp.write_bytes(dump(chapter));pp.write_bytes(dump(project));print(json.dumps({'registered':len(CAPTURES),'headings':sum(n['type']=='heading' for n in content),'images':sum(n['type']=='image' for n in content)}))
if __name__=='__main__':main()
