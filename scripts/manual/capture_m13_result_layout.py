"""Owned hidden Qt screenshots at the formal manual's actual DPR and page zoom."""
import hashlib,json,os,sqlite3,struct,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for part in ('scripts','desktop/src','backend/src','packages/phonetic_core/src'):sys.path.insert(0,str(ROOT/part))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/part) for part in ('scripts','desktop/src','backend/src','packages/phonetic_core/src'))
for key in ('QT_ENABLE_HIGHDPI_SCALING','QT_SCALE_FACTOR','QT_SCREEN_SCALE_FACTORS'):os.environ.pop(key,None)
os.environ['QT_QPA_PLATFORM']='windows'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio --autoplay-policy=no-user-gesture-required'

def main():
    from verification_artifacts import verification_run
    from PyQt6.QtCore import Qt,QEventLoop,QTimer
    from PyQt6.QtWidgets import QApplication,QFileDialog
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    with verification_run('m13-result-layout','manual-qt','Hidden maximized Qt; existing test schema copy; only M13 text, actual DPR1.5/page100, no device or scientific job') as (out,scratch):
        db=scratch/'jobs.sqlite3'
        with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
        cache=scratch/'cache';cache.mkdir();initialize_local_files(cache);register_scheme();app=QApplication(['M13-owned-manual'])
        window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal')
        window.resize(window.screen().geometry().width(),window.screen().geometry().height());window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);window.showMaximized();window.view.setZoomFactor(1);window.page.setAudioMuted(True)
        report={'success':False,'captures':[],'checks':[],'errors':[]}
        export_file=out/'m13-export-kaiti-independent-colors.png'
        QFileDialog.getSaveFileName=lambda *args,**kwargs:(str(export_file),'PNG (*.png)')
        def pause(ms=100):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            wrapped='(()=>{try{return {ok:true,value:eval('+json.dumps(code)+')??null};}catch(e){return {ok:false,error:String(e)};}})()'
            loop,values=QEventLoop(),[];window.page.runJavaScript(wrapped,lambda value:(values.append(value),loop.quit()));QTimer.singleShot(8000,loop.quit);loop.exec()
            if not values:raise RuntimeError('JavaScript timeout')
            if not values[0] or not values[0].get('ok'):raise RuntimeError(str(values[0]))
            return values[0]['value']
        def until(code):
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                if js(code):return
                pause()
            raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-4000)')))
        def click(label):
            name=json.dumps(label)
            expr='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")==='+name+'||b.getAttribute("title")==='+name+'||b.textContent.trim().replace(/\\s*\\*$/," ").trim()==='+name+'))'
            until('!!('+expr+')&&!('+expr+').disabled');js(expr+'.click()');pause(150)
        def input_text(text):
            js('(()=>{const e=document.querySelector("textarea[aria-label=待转换汉字文本]");e.value='+json.dumps(text)+';e.dispatchEvent(new Event("input",{bubbles:true}));})()');pause(180)
        def select(value):
            js('(()=>{const e=document.querySelector("select[aria-label=转换标准]");e.value='+json.dumps(value)+';e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause(180)
        def range_value(label,value):
            js('(()=>{const e=document.querySelector('+json.dumps('input[aria-label="'+label+'"]')+');e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));})()');pause(180)
        def radio(value):
            js('(()=>{const e=document.querySelector('+json.dumps('input[value="'+value+'"]')+');e.checked=true;e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause(180)
        def open_token(index):
            js('document.querySelector('+json.dumps('.m13-output [data-index="'+str(index)+'"]')+').click()');until('!!document.querySelector(".m13-variants")');pause(50)
        def capture(name,popup_index=None):
            until('document.fonts.status==="loaded"');pause(350)
            if popup_index is not None:
                open_token(popup_index)
            image=window.view.grab().toImage();size=[image.width(),image.height()];dpr=image.devicePixelRatio()
            assert abs(dpr-1.5)<.01 and size in ([2560,1600],[2561,1601]),(size,dpr)
            assert window.isMaximized() and window.view.zoomFactor()==1,{'maximized':window.isMaximized(),'zoom':window.view.zoomFactor(),'state':int(window.windowState().value),'window':[window.width(),window.height()]}
            assert js('parseFloat(getComputedStyle(document.documentElement).zoom)||1')==1
            if size!=[2560,1600]:image=image.copy(0,0,2560,1600)
            path=out/(name+'.png');assert image.save(str(path))
            report['captures'].append({'name':name,'file':str(path),'image':[2560,1600],'raw_image':size,'dpr':dpr,'pageZoom':1,'maximized':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});print('Captured',name,flush=True)
        try:
            until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"');window.showMaximized();window.view.setZoomFactor(1);pause(350);click('设置');click('浅色');click('汉字转国际音标')
            js('document.querySelector("[aria-label=收起侧栏]")?.click()');pause(180)
            input_layout=js('(()=>{const section=document.querySelector(".m13-input-section"),input=section.querySelector("textarea").getBoundingClientRect(),result=document.querySelector(".m13-output").getBoundingClientRect();return {countInHeading:!!section.querySelector(".module-section-heading .m13-count"),bottomDifference:input.bottom-result.bottom,leftWidth:section.getBoundingClientRect().width,rightWidth:document.querySelector(".m13-result-section").getBoundingClientRect().width};})()')
            assert input_layout['countInHeading'] and abs(input_layout['bottomDifference'])<.1 and abs(input_layout['leftWidth']-input_layout['rightWidth'])<1,input_layout;report['input_layout']=input_layout
            capture('m13-overview-current-20261011')
            input_text('实验语音学\n重行长乐');capture('m13-convert-flow-light-scaled-r20261007')
            capture('m13-polyphony-detail-r4-light-scaled-r20261007',0);click('关闭读音选择');capture('m13-reading-choice-detail-scaled-r20261007',0);click('关闭读音选择')
            input_text('银行行走');open_token(1)
            js('[...document.querySelectorAll(".m13-variants>button")].find(b=>b.textContent.trim().startsWith("hang2")).click()');pause(180)
            capture('m13-hang-position-choice-r2-light-scaled-r20261007',1);click('关闭读音选择')
            input_text('🙂银行行走');capture('m13-position-edit-r2-light-scaled-r20261007')
            input_text('春江花月夜\n银行行走');select('Standard Chinese (Beijing)严');radio('ipa-only');capture('m13-beijing-strict-onlyipa-r2-light-scaled-r20261007');radio('paired')
            select('赵元任《汉语口语语法》');capture('m13-zhao-comparison-r2-light-scaled-r20261007')
            input_text('春江花月夜\n银行行走');select('UntPhesoca严');capture('m13-unt-current-20261011')
            input_text('花儿女儿门儿\n儿子 花 儿');click('自动组合儿化')
            assert js('document.querySelector(".m13-erhua p").textContent.endsWith("自动组合后请核对独立的儿字。")')
            js('document.querySelectorAll("[data-erhua=true]")[1].click()');pause(180);click('拆分儿化');capture('m13-erhua-current-20261011')
            js('document.querySelector("[data-erhua=true]").click()');pause(180);capture('m13-erhua-split-20261011');click('关闭读音选择')
            select('Standard Chinese (Beijing)');input_text('春江潮水连海平，海上明月共潮生。\n银行行走，花儿门儿。');range_value('汉字字号',26);range_value('音标字号',20);click('等宽排版')
            uniform=js('(()=>{const cells=[...document.querySelectorAll(".m13-token")];const widths=cells.map(e=>e.getBoundingClientRect().width);return {min:Math.min(...widths),max:Math.max(...widths),count:cells.length};})()');assert uniform['max']-uniform['min']<.1;report['qt_uniform']=uniform
            range_value('行距',8);tight=js('document.querySelectorAll(".m13-paragraph")[1].getBoundingClientRect().top');range_value('行距',30);loose=js('document.querySelectorAll(".m13-paragraph")[1].getBoundingClientRect().top');assert loose>tight+40;report['qt_row_shift']=loose-tight
            range_value('行距',18);capture('m13-equal-width-20261011')
            click('等宽排版');input_text('春江花月夜\n银行行走');select('Standard Chinese (Beijing)严');range_value('汉字字号',32);range_value('音标字号',22);range_value('字音间距',4);range_value('行距',22)
            js('(()=>{const e=document.querySelector("input[value=stacked]");e.checked=true;e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause(180)
            js('(()=>{const e=document.querySelector(".font-family-select select");e.value="KaiTi";e.dispatchEvent(new Event("change",{bubbles:true}));const b=[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==="B 粗体");b?.click();for(const [label,color] of [["音标颜色","#2345ab"],["汉字颜色","#ab4523"]]){const c=document.querySelector("input[aria-label="+label+"]");c.value=color;c.dispatchEvent(new Event("input",{bubbles:true}));}})()');pause(250);capture('m13-layout-current-20261011')
            js('window.qaPngPaint=[];(()=>{const original=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(text,...args){window.qaPngPaint.push(text);return original.call(this,text,...args);};})()');click('保存为 PNG')
            deadline=time.monotonic()+30
            while not export_file.is_file() and time.monotonic()<deadline:pause()
            assert export_file.is_file(),js('document.body.innerText.slice(-1000)')
            assert '▼' not in js('window.qaPngPaint')
            data=export_file.read_bytes();assert data[1:4]==b'PNG'
            report['export']={'file':str(export_file),'image':list(struct.unpack('>II',data[16:24])),'sha256':hashlib.sha256(data).hexdigest(),'triangleAbsent':True}
            js('(()=>{const e=document.querySelector("textarea[aria-label=待转换汉字文本]");e.focus();e.select();})()');capture('m13-input-selected-r2-light-scaled-r20261007')
            click('保存本机草稿');click('关闭 汉字转国际音标');click('汉字转国际音标');capture('m13-draft-explicit-restore-r2-light-scaled-r20261007')
            click('恢复本机草稿');input_text('春江花月夜\n银行行走！');click('关闭 汉字转国际音标');capture('m13-unsaved-close-dialog-r2-light-scaled-r20261007');click('取消关闭')
            input_text('春江花月夜\n银行行走')
            js('(()=>{const e=document.querySelector(".font-family-select select");e.value="__custom__";e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause(180)
            js('(()=>{const e=document.querySelector(".font-family-select input");e.value="M13-Missing-Font-QA";e.dispatchEvent(new Event("input",{bubbles:true}));})()');pause(180);click('保存为 PNG');until('(()=>{const s=document.querySelector(".mandarin-ipa-page .module-status");return s?.textContent.includes("M13-Missing-Font-QA")&&s.textContent.includes("不可用");})()');capture('m13-unavailable-font-error-r2-light-scaled-r20261007')
            report['checks']=['equal width in actual Qt','row spacing changes real row location','character count in heading and aligned input/result bottom','hidden Qt DPR1.5 page100 maximized','all M13 screenshots refreshed including draft and error states'];report['success']=True
        except Exception:report['errors'].append(traceback.format_exc());raise
        finally:
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');window.closing=True;window.close();pause(500);app.quit();print('Evidence:',out,flush=True)

if __name__=='__main__':main()
