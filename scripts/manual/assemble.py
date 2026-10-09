"""Register verified screenshots and synchronize the editable manual index.

Chapter authors decide where each image and audio example belongs. This script
never inserts media into a chapter or changes a chapter body.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from core import dump,node_text,read_json,safe_file

ROOT=Path(__file__).resolve().parents[2]
MANUAL=ROOT/'manual'

def walk(node):
    yield node
    for child in node.get('content',[]):yield from walk(child)

def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-report',action='append',type=Path,default=[])
    parser.add_argument('--analysis-report',action='append',type=Path,default=[])
    args=parser.parse_args()
    project=read_json(MANUAL/'project.json');assets={a['id']:a for a in project['assets']}
    registered=[]
    for report_path in args.capture_report:
        report=read_json(report_path)
        if report.get('success') is not True:raise ValueError('Unfinished capture report')
        for capture in report['captures']:
            if capture['id'].startswith('manual-reader'):continue # Re-capture after all chapters are consolidated.
            if not capture.get('maximized'):raise ValueError('Capture was not maximized')
            if not re.fullmatch(r'[a-z0-9-]+',capture['id']):raise ValueError('Invalid capture identity')
            theme=capture.get('theme')
            if capture['id'].endswith('-light') and theme!='light':raise ValueError('Capture theme differs from its identity')
            if capture['id'].endswith('-dark') and theme!='dark':raise ValueError('Capture theme differs from its identity')
            file=Path(capture['file']);raw=file.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest!=capture['sha256']:raise ValueError('Capture changed')
            asset_id=capture['id']
            if asset_id in assets and assets[asset_id]['sha256']!=digest:
                asset_id=asset_id+'-v3-'+digest[:8]
            relative='assets/software-only/screenshots/'+asset_id+'.png'
            target=safe_file(MANUAL,relative);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Capture ID collides with different pixels')
            if not target.exists():shutil.copyfile(file,target)
            caption=re.sub(r'^图\s*[\d -]+[:：]\s*','',capture['caption'])
            assets[asset_id]=dict(id=asset_id,path=relative,kind='image',mime='image/png',sha256=digest,
                caption=caption,width=capture['image'][0],height=capture['image'][1],git=False,distribution='software-only',
                sourceType='实际应用截图',source='Windows 应用内工作台，最大化后捕获；使用作者授权的本地示例副本。')
            registered.append(dict(captureId=capture['id'],assetId=asset_id,chapterId=capture['chapterId']))
    for report_path in args.analysis_report:
        report=read_json(report_path)
        if report.get('success') is not True:raise ValueError('Unfinished analysis report')
        plot=report['exportPlot'];file=Path(plot['file']);raw=file.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=plot['sha256']:raise ValueError('Analysis plot changed')
        asset_id=plot['id']
        if not re.fullmatch(r'[a-z0-9-]+',asset_id):raise ValueError('Invalid plot identity')
        if asset_id in assets and assets[asset_id]['sha256']!=digest:asset_id=asset_id+'-v3-'+digest[:8]
        relative='assets/software-only/plots/'+asset_id+'.png'
        target=safe_file(MANUAL,relative);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Plot ID collides with different pixels')
        if not target.exists():shutil.copyfile(file,target)
        caption=re.sub(r'^图\s*[\d -]+[:：]\s*','',plot['caption'])
        assets[asset_id]=dict(id=asset_id,path=relative,kind='image',mime='image/png',sha256=digest,
            caption=caption,width=plot['image'][0],height=plot['image'][1],git=False,distribution=plot['distribution'],
            sourceType=plot['sourceType'],source='Windows 应用从作者授权录音的本次分析中导出的结果图；DPI 与文件哈希见拍摄报告。')
        chapter_id=report['chapterId'] if 'chapterId' in report else report['captures'][0]['chapterId']
        registered.append(dict(captureId=plot['id'],assetId=asset_id,chapterId=chapter_id))
    project['assets']=list(assets.values())
    index=[]
    for descriptor in project['chapters']:
        chapter=read_json(MANUAL/descriptor['path']);body=chapter['body']['content'];cid=chapter['id']
        descriptor['sections']=[dict(id=n['attrs']['id'],title=node_text(n),level=n['attrs']['level']) for n in walk(chapter['body']) if n['type']=='heading']
        for n in body:
            target=n.get('attrs',{}).get('id');text=node_text(n)
            if text:index.append(dict(chapterId=cid,**({'targetId':target} if target else {}),text=text,title=descriptor['title']))
    project['searchIndex']=index;project['softwareVersion']='3.0.0-preview.1'
    (MANUAL/'project.json').write_bytes(dump(project))
    print(json.dumps(dict(registered=registered,assets=len(assets),searchEntries=len(index)),ensure_ascii=False))

if __name__=='__main__':main()
