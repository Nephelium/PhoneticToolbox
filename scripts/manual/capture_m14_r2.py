"""M14-R2 native screenshots from the existing in-workspace author table."""
import argparse,base64,hashlib,json,os,sqlite3,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for part in ('scripts','desktop/src','backend/src','packages/phonetic_core/src'):sys.path.insert(0,str(ROOT/part))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/part) for part in ('scripts','desktop/src','backend/src','packages/phonetic_core/src'))
for name in ('QT_ENABLE_HIGHDPI_SCALING','QT_SCALE_FACTOR','QT_SCREEN_SCALE_FACTORS'):os.environ.pop(name,None)
os.environ['QT_QPA_PLATFORM']='windows'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio --autoplay-policy=no-user-gesture-required'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    from verification_artifacts import verification_run
    from PyQt6.QtCore import Qt,QEventLoop,QTimer
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);args=parser.parse_args()
    source=args.source.resolve(strict=True);original=sha(source)
    with verification_run('m14','r2-manual-qt','Existing in-workspace author table; start row 1; native maximized Qt DPR 1.5 and page 100%') as (out,scratch):
        db=scratch/'jobs.sqlite3'
        with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as template,sqlite3.connect(db) as target:template.backup(target)
        cache=scratch/'cache';cache.mkdir();initialize_local_files(cache);register_scheme();app=QApplication(['M14-R2-owned-manual'])
        window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal')
        window.resize(window.screen().geometry().width(),window.screen().geometry().height());window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);window.showMaximized();window.view.setZoomFactor(1.);window.page.setAudioMuted(True)
        report=dict(success=False,source_sha256=original,captures=[],checks=[],errors=[],scope='Windows native Qt, in-workspace author table, no schema migration or physical device test')
        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda v: (values.append(v), loop.quit()))
            QTimer.singleShot(8000, loop.quit)
            loop.exec()
            if not values:
                raise RuntimeError('JavaScript timeout')
            return values[0]

        def until(code, seconds=90):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-4000)')))

        def click(label):
            expr = '[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")||b.textContent.trim())===' + json.dumps(label) + ')'
            until('!!(' + expr + ')&&!(' + expr + ').disabled')
            js(expr + '.click()')
            pause(120)

        def select(label, value):
            assert js('(()=>{const e=[...document.querySelectorAll("select")].find(e=>e.offsetParent&&(e.getAttribute("aria-label")===' + json.dumps(label) + '||e.closest("label")?.textContent.trim().startsWith(' + json.dumps(label) + ')));if(!e||![...e.options].some(o=>o.value===' + json.dumps(str(value)) + '))return false;e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), label
            pause(150)

        def fill(label, value):
            assert js('(()=>{const e=document.querySelector(' + json.dumps('input[aria-label="' + label + '"]') + ');if(!e)return false;e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), label
            pause(150)

        def idle():
            until('document.querySelector(".phonology-page")?.getAttribute("aria-busy")==="false"')

        def upload(path):
            raw = base64.b64encode(path.read_bytes()).decode()
            js('(()=>{const d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob(' + json.dumps(raw) + '),c=>c.charCodeAt(0))],' + json.dumps(path.name) + '));const e=document.querySelector(".phonology-page input[type=file]");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            idle()
            until('!!document.querySelector("table[aria-label=原始表格样例]")')


        def capture(name):
            until("document.fonts.status==='loaded'");pause(450);image=window.view.grab().toImage();size=[image.width(),image.height()]
            assert abs(image.devicePixelRatio()-1.5)<.01 and size in ([2560,1600],[2561,1601]),(size,image.devicePixelRatio())
            if size!=[2560,1600]:image=image.copy(0,0,2560,1600)
            path=out/(name+'.png');assert image.save(str(path));report['captures'].append(dict(name=name,file=str(path),image=[2560,1600],raw_image=size,dpr=image.devicePixelRatio(),pageZoom=window.view.zoomFactor(),maximized=window.isMaximized(),theme=js('document.documentElement.dataset.theme'),sha256=sha(path)));print('Captured '+name,flush=True)
        try:
            until("!!document.querySelector('nav')&&document.fonts.status==='loaded'");click('设置');click('浅色');click('音系归纳');js("document.querySelector('[aria-label=\"收起侧栏\"]')?.click()");upload(source);fill('数据开始行',1);idle()
            assert js("(()=>{const c=document.querySelector('.workbench-center');return c.scrollHeight<=c.clientHeight+1})()")
            capture('import');click('确认导入并继续');idle();until("document.querySelector('.phonology-page').innerText.includes('5453 条记录')")
            for tone,name in {'1':'阴平（暂拟）','2':'阳平（暂拟）','5':'阴去（暂拟）','6':'阳去（暂拟）','7':'阴入（暂拟）','8':'阳入（暂拟）'}.items():fill('调类 '+tone,name)
            click('保存调类设置并继续');click('保存声韵设置并继续');capture('initial');click('韵母 → 声母DOCX');js("document.querySelector('.word-sheet').scrollTop=document.querySelector('.summary').offsetHeight+70");capture('final');click('二维声韵表XLSX');until("document.querySelectorAll('.matrix-scroll tbody tr').length===12");capture('matrix');click('生成三份结果');idle();until("document.querySelector('.phonology-page').innerText.includes('三份结果已完整生成')");click('声母 → 韵母DOCX');capture('result')
            fill('搜索字头或 IPA','phoŋ1');until("!!document.querySelector('.preview-entry.focused')");click('二维声韵表XLSX');until("!!document.querySelector('.matrix-sheet .preview-entry.focused')")
            assert sha(source)==original;report['checks']=['5453 source rows retained','native 2560x1600 DPR 1.5 light page100 maximized','import actions fit, two Word previews, matrix and generation','search moves between both document and matrix views','source SHA unchanged'];report['success']=True
        except Exception:report['errors'].append(traceback.format_exc());raise
        finally:
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');window.close();pause(500);app.quit();print(json.dumps({'success':report['success'],'report':str(out/'report.json')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
