"""Actual M15 design, synthetic ABX sessions and export in an owned Qt profile.

Experiment instructions and answers below are explicit demonstration material.
The UI language switch never rewrites these stored materials or historical RTs.
"""
from __future__ import annotations
import argparse,base64,hashlib,io,json,math,struct,sys,time,wave
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from manual.capture_a_support import Capture
from verification_artifacts import verification_run


def run(args,out,scratch):
    c=Capture(args,out,scratch,'m15','perception')
    from PyQt6.QtCore import QPoint,Qt,QTimer
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QFileDialog,QMessageBox
    downloads=[]
    def destination(parent,title,suggested,file_filter,**kwargs):
        target=scratch/(str(len(downloads))+'-'+Path(suggested).name)
        downloads.append(target)
        return str(target),file_filter
    QFileDialog.getSaveFileName=destination
    # Confirm only dialogs in this isolated process, for the owned fixture edits.
    timer=QTimer()
    def confirm():
        for dialog in c.app.topLevelWidgets():
            if isinstance(dialog,QMessageBox) and dialog.isVisible():
                button=dialog.button(QMessageBox.StandardButton.Yes)
                if button:button.click()
    timer.timeout.connect(confirm);timer.start(100)
    def label(text):return c.ui.label(text)
    def selector(text):
        return '[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(label(text),ensure_ascii=False)+')'
    def click(text):
        expr=selector(text);c.until('!!('+expr+')&&!('+expr+').disabled');c.js(expr+'.click()');c.pause()
    def pointer(text):
        expr=selector(text);c.until('!!('+expr+')&&!('+expr+').disabled')
        c.js(expr+'.scrollIntoView({block:"nearest"})');c.pause(80)
        r=c.js('(()=>{const r='+expr+'.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
        target=c.window.view.focusProxy() or c.window.view
        pos=target.mapFromGlobal(c.window.view.mapToGlobal(QPoint(round(r[0]),round(r[1]))))
        QTest.mouseClick(target,Qt.MouseButton.LeftButton,pos=pos);c.pause(80)
    def field(text,value,event='input'):
        c.fill('[aria-label='+json.dumps(label(text),ensure_ascii=False)+']',value,event)
    def phase(value):c.until('document.querySelector(".m15-run")?.dataset.phase==='+json.dumps(value),30)
    def idle():c.until('document.querySelector(".perception-page")?.getAttribute("aria-busy")==="false"')
    def toggle(text,wanted):
        assert c.js('(()=>{const l=[...document.querySelectorAll("label")].find(e=>e.textContent.trim()==='+json.dumps(label(text),ensure_ascii=False)+'),e=l?.querySelector("input");if(!e||e.disabled)return false;if(e.checked!=='+str(wanted).lower()+')e.click();return true})()'),text
        c.pause()
    def upload(selector,files):
        encoded=[dict(name=name,data=base64.b64encode(data).decode(),mime=mime) for name,data,mime in files]
        c.js('(()=>{const d=new DataTransfer();for(const f of '+json.dumps(encoded)+'){const a=Uint8Array.from(atob(f.data),x=>x.charCodeAt(0));d.items.add(new File([a],f.name,{type:f.mime}));}const e=document.querySelector('+json.dumps(selector,ensure_ascii=False)+');e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()');idle()
    def stimulus(role,index):
        b=io.BytesIO()
        with wave.open(b,'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(48000)
            frequency={'A':180,'B':260,'X':220}[role]+index*13
            stream.writeframes(b''.join(struct.pack('<h',round(1000*math.sin(2*math.pi*frequency*i/48000))) for i in range(72000)))
        return (role+str(index)+'-demo.wav',b.getvalue(),'audio/wav')
    def import_role(role,indices):
        aria=c.ui.key('modules.perception.text.3634ba595a67')+role
        upload('[aria-label='+json.dumps(aria,ensure_ascii=False)+']',[stimulus(role,i) for i in indices])
    def shot(name):
        full='m15-'+name
        if not args.scene or full in args.scene:c.shot(full)
    def key(value):QTest.keyClick(c.window.view.focusProxy(),value);c.pause(80)
    try:
        c.until('!!document.querySelector(".perception-page")');c.prepare();idle()
        field('实验范式','ABX','change');shot('abx-paradigm-r2')
        field('实验范式','X','change');import_role('X',[1]);shot('overview-light')
        click('参数');shot('parameters-detail-r4-light')
        field('推进方式','manual','change');toggle('自动播放',False);toggle('提示音',False)
        field('ISI 毫秒',300);field('试次间隔毫秒',1000)
        field('实验标题','合成 ABX 操作演示')
        field('指导语','操作演示：请听完 A、B、X 后按 J 或 F 判断。')
        c.fill('.m15-form label:nth-child(3) textarea','操作演示：准备好后进入下一试次。')
        c.fill('.m15-design-panel > .m15-row input[maxlength="1"]','j')
        keys=c.js('[...document.querySelectorAll(".m15-design-panel > .m15-row input[maxlength]")].length');assert keys==2
        c.fill('[aria-label='+json.dumps(c.ui.key('modules.perception.text.a8c36c98b90f')+'1',ensure_ascii=False)+']','X 与 A 相同')
        c.fill('[aria-label='+json.dumps(c.ui.key('modules.perception.text.1e7af96249e5')+'2',ensure_ascii=False)+']','f')
        c.fill('[aria-label='+json.dumps(c.ui.key('modules.perception.text.a8c36c98b90f')+'2',ensure_ascii=False)+']','X 与 B 相同')
        shot('parameters-manual-r2')
        field('实验范式','ABX','change')
        click('添加按键分段')
        c.fill('.m15-design-panel section.m15-group input[type=number]',1)
        c.fill('.m15-design-panel section.m15-group .m15-row label:nth-child(2) input',2)
        c.fill('.m15-design-panel section.m15-group textarea','操作演示阶段：J 表示 X 与 A 相同，F 表示 X 与 B 相同。')
        for key_value,meaning in [('j','X 与 A 相同'),('f','X 与 B 相同')]:
            click('添加分段按键')
            c.fill('.m15-design-panel section.m15-group .m15-row:last-of-type input:first-child',key_value)
            c.fill('.m15-design-panel section.m15-group .m15-row:last-of-type input:nth-child(2)',meaning)
        shot('key-range-r2')
        click('素材');import_role('A',[1,2]);import_role('B',[1]);import_role('X',[2])
        click('序列');pointer('预检与试音');idle();c.until('!!document.querySelector(".module-status.error")||!!document.querySelector("[role=alert]")');shot('sequence-missing-r2')
        click('素材');import_role('B',[2]);shot('abx-assets-r2')
        c.js('[...document.querySelectorAll(".m15-asset")].find(e=>e.textContent.includes("A1-demo.wav")).querySelector("button").click()');idle();c.until('!!document.querySelector(".m15-preview audio")');shot('audio-preview-r2');click('停止预览')
        click('序列')
        for i in (1,2):
            c.fill('.m15-table tbody tr:nth-child('+str(i)+') input','操作演示：第 '+str(i)+' 试次，听完后按 J/F。')
        c.js('document.querySelector(".workbench-right-heading button").click();document.querySelector(".m15-table").scrollLeft=9999');c.pause();shot('sequence-right-columns-r2')
        c.js('document.querySelector(".workbench-right-heading button").click();document.querySelector(".m15-table").scrollLeft=0')
        click('添加洗牌范围');c.fill('.m15-design-panel > .m15-row:nth-of-type(2) label:nth-child(1) input',1)
        c.fill('.m15-design-panel > .m15-row:nth-of-type(2) label:nth-child(2) input',2);shot('sequence-linked-r2')
        click('问卷');click('添加问卷题目')
        c.fill('.m15-design-panel > .m15-group:nth-child(3) .m15-row label:first-child input','语言背景')
        c.fill('.m15-design-panel > .m15-group:nth-child(3) select','checkbox','change')
        # Add two explicit demonstration options through their real controls.
        for value in ('普通话','其他语言'):
            c.js('document.querySelector(".m15-design-panel > .m15-group:nth-child(3) .m15-options > button").click()');c.pause()
            c.fill('.m15-design-panel > .m15-group:nth-child(3) .m15-options .m15-row:last-of-type input',value)
        shot('questionnaire-design-r2')
        click('资源与恢复');field('路径前缀','demo-stimuli')
        upload('[aria-label='+json.dumps(label('资源助手目录'),ensure_ascii=False)+']',[stimulus('A',1),stimulus('B',1),stimulus('X',1)])
        c.until('true');shot('resource-helper-r2')
        before=len(downloads);click('导出序列 XLSX');c.until('true')
        deadline=time.monotonic()+15
        while time.monotonic()<deadline and not (len(downloads)>before and downloads[-1].is_file()):c.pause()
        assert len(downloads)>before and downloads[-1].is_file()
        sequence=downloads[-1].read_bytes();upload('[aria-label='+json.dumps(label('导入序列 XLSX'),ensure_ascii=False)+']',[('demo-sequence.xlsx',sequence,'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')]);shot('sequence-import-feedback-r2')
        pointer('预检与试音');idle();toggle('已听见试音，设备正确',True);toggle('已暂停其他录制、播放及重型任务',True);shot('preflight-r2')
        click('填写被试信息');c.until('!!document.querySelector(".m15-questionnaire input")');shot('participant-empty-r2')
        c.fill('.m15-questionnaire input','DEMO-P01')
        c.js('document.querySelector(".m15-questionnaire input[type=radio]").click();document.querySelector(".m15-questionnaire input[type=checkbox]").click()');pointer('建立独立会话');phase('intro');shot('intro-r2')
        pointer('开始实验');phase('notice');shot('phase-notice-r2')
        pointer('继续阶段');phase('interstitial');shot('trial-instruction-r2')
        pointer('进入试次');phase('ready');shot('manual-play-ready-r2')
        pointer('播放当前试次');phase('playing');shot('abx-playing-r2')
        phase('responding');shot('response-open-r2');key(Qt.Key.Key_J)
        phase('interstitial');pointer('进入试次');phase('ready');pointer('播放当前试次');phase('playing');pointer('暂停');phase('paused');shot('paused-r2')
        pointer('明确继续 / 重新呈现');phase('interstitial');pointer('进入试次');phase('ready');pointer('播放当前试次');phase('responding');key(Qt.Key.Key_F);phase('completed');idle();shot('completed-r2')
        field('结果格式','json','change');before=len(downloads);click('导出结果');idle()
        deadline=time.monotonic()+15
        while time.monotonic()<deadline and not (len(downloads)>before and downloads[-1].is_file()):c.pause()
        result=json.loads(downloads[-1].read_text(encoding='utf-8'))
        assert [a['status'] for a in result['attempts']]==['completed','interrupted','completed']
        assert result['project']['trials'][0]['instruction']=='操作演示：第 1 试次，听完后按 J/F。'
        original_attempts=json.dumps(result['attempts'],ensure_ascii=False,sort_keys=True)
        click('确认导出文件已保存');idle();click('结束实验');idle();shot('recent-result-r2')
        before=len(downloads);click('查看结果');phase('completed');idle();shot('terminal-recovery-r2');assert len(downloads)==before
        c.report.update(success=True,checks=['Six synthetic WAVs imported through role-specific File controls; missing B blocks preflight','Real media preview, manual design, XLSX export/reimport and required questionnaire','Actual native Qt ABX playback, response keys and pause/re-presentation','JSON readback: two complete responses, one interrupted attempt; original instructions retained','Terminal recovery does not replay or automatically export'],physicalTimingMeasured=False,downloads=[p.name for p in downloads],attemptsSha256=hashlib.sha256(original_attempts.encode()).hexdigest())
    finally:
        timer.stop();c.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--locale',choices=('en','ja'),required=True);parser.add_argument('--frontend-dist',type=Path,default=ROOT/'frontend/dist');parser.add_argument('--scene',action='append',default=[]);args=parser.parse_args();args.frontend_dist=args.frontend_dist.resolve(strict=True)
    with verification_run('m15-localized',args.locale,'Actual owned Qt, synthetic ABX stimuli, native playback/key events, preserved experiment material and JSON readback') as (out,scratch):run(args,out,scratch)
