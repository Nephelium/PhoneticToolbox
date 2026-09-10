"""M01-F2 real Qt dialog, shared buttons, actual worker, files and normal close."""
import json
import hashlib
import io
import sqlite3
from pathlib import Path
import time
from uuid import uuid4
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog,QLineEdit
from ptb_desktop.host import register_scheme,Workbench
from ptb_worker.local_acoustic_files import initialize_local_files
from baseline_support import create_fixture,RECIPES
from m01_legacy_fixtures import create_legacy_fixtures

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'output/validation/m01'/('task-window-'+uuid4().hex);folder.mkdir()
    inputs=folder/'输入';inputs.mkdir();cache=folder/'cache';cache.mkdir();initialize_local_files(cache)
    wav=create_fixture(inputs,RECIPES[0])
    (inputs/(wav.stem+'.TextGrid')).write_text('File type = "ooTextFile short"\n"TextGrid"\n0\n.8\n<exists>\n1\n"IntervalTier"\n"音节"\n0\n.8\n2\n0\n.4\n"阴平"\n.4\n.8\n"上声"\n',encoding='utf-8')
    legacy=create_legacy_fixtures(inputs)
    original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    register_scheme();app=QApplication(['PhoneticToolbox']);window=Workbench(ROOT/'frontend/dist',test=True,
        jobs_path=ROOT/'output/validation/p06/local-state.sqlite3',local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe');window.show()
    result={'success':False,'stages':[]};start=time.monotonic();stage=0;pending=False
    click=lambda label:"[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==="+json.dumps(label,ensure_ascii=False)+")?.click()"
    def select(label,value):return "(()=>{const e=document.querySelector('select[aria-label=\""+label+"\"]');e.value=[...e.options].find(o=>o.textContent==="+json.dumps(value,ensure_ascii=False)+").value;e.dispatchEvent(new Event('change',{bubbles:true}));})()"
    def picker():
        dialog=app.activeModalWidget()
        if not isinstance(dialog,QFileDialog):QTimer.singleShot(100,picker);return
        dialog.setDirectory(str(inputs.parent))
        def accept():
            if time.monotonic()-start>100:return
            edit=dialog.findChild(QLineEdit,'fileNameEdit')
            if edit:edit.setText(str(inputs))
            selected=dialog.selectedFiles()
            if len(selected)==1 and Path(selected[0]).resolve()==inputs.resolve():dialog.accept()
            else:QTimer.singleShot(150,accept)
        QTimer.singleShot(150,accept)
    checks=[
        ("document.querySelector('.host-badge')?.textContent==='本地桌面'",click('参数估计')),
        ("!!document.querySelector('.m01-page')",click('选择音频目录')),
        ("document.querySelectorAll('.m01-file-list .file-row').length===1","document.querySelector('.m01-file-list .file-row')?.click()"),
        ("document.querySelectorAll('.m01-intervals button').length===2","document.querySelector('.association-help').open=true"),
        ("document.querySelector('.association-help')?.open",select('旧唇形PKL','旧唇形.pkl')),
        ("!!document.querySelector('select[aria-label=\"旧唇形PKL\"]')?.value",click('转换并保存 .lip.json')),
        ("document.querySelector('.notice')?.textContent.includes('已保存 旧唇形.lip.json') && [...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='开始全列表分析'&&!b.disabled)",click('开始全列表分析')),
        ("document.querySelector('.m01-results strong')?.textContent==='1 / 1 已完成' && document.querySelector('.m01-page')?.textContent.includes('已保存3个结果文件') && !!document.querySelector('.m01-tiers input[type=checkbox]')","document.querySelector('.m01-tiers input[type=checkbox]')?.click()"),
        ("document.querySelector('.m01-tiers input[type=checkbox]')?.checked && [...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='保存当前层切分音频'&&!b.disabled)",click('保存当前层切分音频')),
        ("document.querySelector('.notice')?.textContent.includes('已保存7个结果文件')",select('切分参数来源','指定历史参数表（来源未核实）')),
        ("document.querySelector('select[aria-label=\"切分参数来源\"]')?.value==='legacy'",select('历史参数表关联','历史参数.xlsx')),
        ("document.querySelector('select[aria-label=\"历史参数表关联\"]')?.selectedOptions[0]?.textContent==='历史参数.xlsx'",click('保存当前层切分音频')),
        ("document.querySelector('.notice')?.textContent.includes('已保存7个结果文件')",select('历史参数表关联','历史参数.ptb.sqlite')),
        ("document.querySelector('select[aria-label=\"历史参数表关联\"]')?.selectedOptions[0]?.textContent==='历史参数.ptb.sqlite'",click('保存当前层切分音频')),
        ("document.querySelector('.notice')?.textContent.includes('已保存7个结果文件')",None)]
    def finish(ok,error=None):
        timer.stop();result.update(success=ok,error=error)
        if not ok:window.view.grab().save(str(folder/'failed.png'));window.provider.close();window.service.close();window.closing=True
        window.close()
        QTimer.singleShot(5000,lambda:app.exit(1))
    def received(ok):
        nonlocal stage,pending
        pending=False
        if not ok:return
        if stage==7 and not (inputs/(wav.stem+'.ptb.json')).exists():return
        if stage in (9,12,14) and len(list(inputs.glob('*.ptb.sqlite')))<{9:4,12:6,14:8}[stage]:return
        result['stages'].append(stage);action=checks[stage][1];stage+=1
        if stage==2:QTimer.singleShot(100,picker)
        if action:window.page.runJavaScript(action)
        else:
            timer.stop();window.page.runJavaScript("document.documentElement.dataset.theme='light'",lambda _:QTimer.singleShot(200,light))
    def light():
        window.view.grab().save(str(folder/'qt-light.png'));window.page.runJavaScript("document.documentElement.dataset.theme='dark'",lambda _:QTimer.singleShot(200,dark))
    def dark():
        window.view.grab().save(str(folder/'qt-dark.png'));window.resize(1000,760);QTimer.singleShot(300,narrow)
    def narrow():
        window.view.grab().save(str(folder/'qt-narrow.png'));finish(True)
    def tick():
        nonlocal pending
        if time.monotonic()-start>180:finish(False,'timeout at stage '+str(stage));return
        if pending or stage>=len(checks):return
        pending=True;window.page.runJavaScript(checks[stage][0],received)
    timer=QTimer();timer.timeout.connect(tick);timer.start(150)
    code=app.exec();result['normal_close']=window.closing and window.provider.closed and window.service.exit_code==0 and code==0
    result['success']=result['success'] and result['normal_close'];result['outputs']=[p.name for p in inputs.iterdir()]
    if result['success']:
        from ptb_worker.io.lip import decode_lip
        converted=decode_lip((inputs/'旧唇形.lip.json').read_bytes())
        assert converted['metadata']=={'audio_first_frame_time':100.,'lip_manual_offset':.025}
        assert all(hashlib.sha256((inputs/n).read_bytes()).hexdigest()==h for n,h in original.items())
        manifests=[json.loads(p.read_text('utf-8')) for p in inputs.glob('*segments.ptb.json')]
        history=[m for m in manifests if m.get('legacy_result')]
        assert {m['legacy_result']['name'] for m in history}=={'历史参数.xlsx','历史参数.ptb.sqlite'}
        for manifest in history:
            ref=manifest['legacy_result'];assert ref['provenance']=='user_associated_unverified' and ref['sha256']==original[ref['name']]
            assert manifest['parent_result_sha256'] is None and not manifest['reestimated']
        matching=0
        for file in inputs.glob('*.ptb.sqlite'):
            if file.name=='历史参数.ptb.sqlite':continue
            with sqlite3.connect(file.as_uri()+'?mode=ro',uri=True) as db:
                columns=[r[1] for r in db.execute('PRAGMA table_info(params)')]
                if columns==['Time_s','Source_Time_s','pF0','textgrid_音节']:
                    rows=db.execute('SELECT * FROM params ORDER BY rowid').fetchall()
                    if len(rows)==2:assert rows==[(0.,0.,120.,'ɑ̃˥'),(.1,.1,None,'')]
                    else:assert rows==[(0.,.4,float('inf'),'ʔ'),(.6-.4,.6,float('-inf'),'β'),(.79-.4,.79,140.,'上声')]
                    matching+=1
        assert matching==4
        result['legacy_verification']=dict(**legacy,preserved_inputs=True,paired_old_formats_via_ui=True,parameter_cut_tables_exact=matching,lip_alignment_exact=True)
    (folder/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(folder/'report.json'),**result),ensure_ascii=False),flush=True)
    window.page.deleteLater();app.processEvents();return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
