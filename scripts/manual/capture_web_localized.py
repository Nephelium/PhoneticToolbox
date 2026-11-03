"""Six real local-Web guide scenes in an owned build/profile and locale."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from manual.capture_locale import CaptureLocale
from verification_artifacts import verification_run
from PIL import Image

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--locale',required=True,choices=['en','ja']);args=parser.parse_args()
 ui=CaptureLocale(args.locale,('account','workbench','components','common'))
 labels=['注册账号','创建研究者账号','返回登录','登录名','密码','登录','文件与空间','项目文件','账号文件空间','任务记录','暂无任务','关闭对话框','容量与扩容','申请总额度（MB）','设置','浅色','知道了']
 with verification_run('web-manual-localized',args.locale,'Original Web six-scene recipe, actual Vue + owned visible Chrome/local HTTP; synthetic account and empty project only') as (out,scratch):
  print(out,flush=True)
  files=[Path(__file__),Path(__file__).with_suffix('.cjs'),ROOT/'scripts/manual/capture_locale.py',ROOT/'scripts/verify_web_manual.py',ROOT/'tests/e2e/preferences-fixture.cjs',ROOT/'manual/chapters/web-introduction.json',ROOT/'frontend/src/main.ts',ROOT/'frontend/src/app/AppShell.vue']
  for folder in ['frontend/src/account','frontend/src/components','frontend/src/design','frontend/src/state','frontend/src/i18n']:
   files.extend(p for p in (ROOT/folder).rglob('*')if p.is_file() and p.suffix in ('.ts','.vue','.json','.css'))
  before={str(p.resolve()):{'path':str(p.resolve()),'sha256':digest(p)}for p in files}
  (out/'config.json').write_text(json.dumps({'locale':args.locale,'labels':{label:ui.label(label)for label in labels}},ensure_ascii=False,indent=2),encoding='utf-8')
  with (out/'build.log').open('w',encoding='utf-8') as log:subprocess.run(['node',str(ROOT/'frontend/node_modules/vite/bin/vite.js'),'build','--outDir',str(scratch/'frontend')],cwd=ROOT/'frontend',stdout=log,stderr=subprocess.STDOUT,check=True)
  # Durable digest of the immutable generated build actually used by this run.
  build={p.relative_to(scratch/'frontend').as_posix():digest(p)for p in (scratch/'frontend').rglob('*')if p.is_file()}
  (out/'build-manifest.json').write_text(json.dumps(build,indent=2),encoding='utf-8')
  with(out/'browser.log').open('w',encoding='utf-8')as log:subprocess.run(['node',str(Path(__file__).with_suffix('.cjs')),str(out),str(scratch)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  report=json.loads((out/'browser.json').read_text('utf-8'));report.update(success=True,sourceChapter='web-introduction',sources=before,sourceChangesDuringRun=[name for name,item in before.items()if digest(Path(name))!=item['sha256']])
  for item in report['captures']:
   with Image.open(out/(item['name']+'-raw.png'))as raw:
    assert raw.size in ((2560,1600),(2561,1601)),raw.size
    item.update(raw_image=list(raw.size),image=[2560,1600]);raw.crop((0,0,2560,1600)).save(out/item['file'])
   item['sha256']=digest(out/item['file'])
  assert report['sourceChangesDuringRun']==[],report['sourceChangesDuringRun']
  (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
  print(json.dumps({'report':str(out/'report.json'),'captures':len(report['captures']),'success':True}),flush=True)
if __name__=='__main__':main()
