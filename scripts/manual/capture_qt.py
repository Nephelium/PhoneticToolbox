"""Capture the owned, maximized Windows workbench for the manual.

Uses private copies of author-approved recordings and a disposable task database.
No camera/microphone capture, global installs, source edits, or public upload.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[2]
DIRS=('desktop/src','backend/src','packages/phonetic_core/src','scripts')
for directory in DIRS:sys.path.insert(0,str(ROOT/directory))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/d) for d in DIRS)
os.environ.setdefault('QT_QPA_PLATFORM','windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio --autoplay-policy=no-user-gesture-required')
egg=ROOT/'.venv/m03-compatible/python.exe'
if egg.is_file():os.environ['PTB_EGG_PYTHON']=str(egg)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--integration-only',action='store_true');parser.add_argument('--rich',action='store_true');parser.add_argument('--details-only',action='store_true');parser.add_argument('--overview-only',action='store_true');parser.add_argument('--flow-only',action='store_true');parser.add_argument('--m02-plot-only',action='store_true');parser.add_argument('--getting-started-only',action='store_true');args=parser.parse_args()
    from PyQt6.QtCore import QEventLoop,QTimer,Qt
    from PyQt6.QtWidgets import QApplication,QFileDialog
    from ptb_desktop.host import Workbench,register_scheme
    from verify_m08_wiring import setup
    out,db,cache=setup();captures=out/'manual-captures';captures.mkdir()
    config=json.loads((ROOT/'local-data/manual-authoring/sources.json').read_text(encoding='utf-8'))
    inputs={}
    for key in ['single','sentence','egg']:
        folder=out/key;folder.mkdir();inputs[key]=folder
        source=Path(config[key]);dest=folder/({'single':'single-syllable','sentence':'speech-sentence','egg':'egg-stereo'}[key]+'.wav')
        shutil.copyfile(source,dest)
        if key=='single':shutil.copyfile(config['target'],folder/'target-syllable.wav')
        for ext in ('.TextGrid','.lab','.xlsx','.ptb.sqlite'):
            sidecar=source.with_suffix(ext)
            if sidecar.is_file():shutil.copyfile(sidecar,dest.with_suffix(ext))
    choice={'folder':inputs['single']}
    QFileDialog.getExistingDirectory=lambda *a,**k:str(choice['folder'])
    QFileDialog.getSaveFileName=lambda *a,**k:(str(out/Path(a[2]).name),'')
    register_scheme();app=QApplication(['PTB-owned-manual-authoring'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal',reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.showMaximized()
    report={'success':False,'scope':'Windows actual Qt; maximized owned hidden window; physical compositor/DPI not claimed',
            'checks':[],'captures':[],'prepared':[],'terminations':[],'out':str(out)}
    w.page.renderProcessTerminated.connect(lambda s,c:report['terminations'].append([s.name,c]))
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda r:(values.append(r),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        if not values:raise RuntimeError('JavaScript timeout')
        return values[0]
    def until(code,seconds=40):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-1000)')))
    def click(text,required=True):
        ok=js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+');if(!e||e.disabled)return false;e.click();return true})()')
        if required and not ok:raise AssertionError('Button unavailable: '+text)
        pause(100);return ok
    def select(label,index=1):
        selector=json.dumps('select[aria-label="'+label+'"]')
        until('document.querySelector('+selector+')?.options.length>'+str(index))
        js('(()=>{const e=document.querySelector('+selector+');e.value=e.options['+str(index)+'].value;e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause(300)
    def fill(label,value):
        selector=json.dumps('[aria-label="'+label+'"]')
        assert js('(()=>{const e=document.querySelector('+selector+')||[...document.querySelectorAll("label")].find(e=>e.offsetParent&&e.textContent.trim().startsWith('+json.dumps(label)+'))?.querySelector("input");if(!e)return false;e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'),label
        pause(100)
    def upload(selector,path):
        raw=base64.b64encode(path.read_bytes()).decode()
        assert js('(()=>{const e=document.querySelector('+json.dumps(selector)+');if(!e)return false;const d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob('+json.dumps(raw)+'),c=>c.charCodeAt(0))],'+json.dumps(path.name)+'));e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));return true})()')
        pause(500)
    def module(mid):
        js('(()=>{const e=[...document.querySelectorAll(".nav-item")].find(e=>e.title==='+json.dumps(titles[mid])+');if(!e)throw Error("module");e.click();})()')
        until('document.querySelector("#tab-'+mid+'")?.getAttribute("aria-selected")==="true"');pause(400)
    def snapshot(cid,name,caption,distribution='software-only'):
        assert w.isMaximized(),'Every capture requires a maximized window'
        pause(350);w.view.repaint();w.view.grab();pause(300);app.processEvents()
        file=captures/(name+'.png');pix=w.view.grab()
        if not pix.save(str(file)):raise RuntimeError('PNG save failed')
        report['captures'].append({'id':name,'chapterId':cid,'file':str(file),'caption':caption,'distribution':distribution,
            'maximized':w.isMaximized(),'window':[w.width(),w.height()], 'frame':[w.frameGeometry().width(),w.frameGeometry().height()],
            'image':[pix.width(),pix.height()],'devicePixelRatio':pix.devicePixelRatio(),
            'theme':js('document.documentElement.dataset.theme'),'palette':js('document.documentElement.dataset.palette'),
            'fontSize':js('getComputedStyle(document.documentElement).fontSize'),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        print('Captured '+name,flush=True)
    titles={'M01':'参数估计','M02':'参数显示','M03':'EGG 信号分析','M04':'LPC 谱图','M05':'唇形提取',
            'M06':'声学参数合成','M07':'发声类型合成','M08':'变速变调','M09':'语谱图转音频',
            'M11':'MFA 自动标注','M12':'TextGrid标注','M13':'汉字转国际音标','M14':'音系归纳',
            'M15':'感知实验','M16':'录音','M17':'国际音标表Plus'}
    try:
        until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"');pause(800)
        assert w.isMaximized();report['checks'].append('Owned Windows Qt workbench is maximized before capture')
        click('设置');click('浅色');until('document.documentElement.dataset.theme==="light"')
        if args.getting_started_only:
            click('首页');until('!!document.querySelector(".home-page")')
            snapshot('getting-started','home-current-light','工作台首页：三组工具入口、左侧导航与底部设置、使用说明、检查更新入口。','public')
            click('使用说明');until('!!document.querySelector(".manual-document")')
            snapshot('getting-started','manual-current-light','应用内使用说明：左侧章节和小节目录、右侧带编号的正文及搜索入口。','public')
            click('设置');click('深色');until('document.documentElement.dataset.theme==="dark"')
            click('使用说明');until('!!document.querySelector(".manual-document")')
            snapshot('getting-started','manual-current-dark','深色主题下的应用内说明书；目录、正文、分隔线和强调色随主题切换。','public')
            assert not report['terminations'];report['success']=True
            return
        click('使用说明');until('!!document.querySelector(".manual-document")');until('document.querySelector(".manual-chapter-header h1")?.textContent.includes("开始使用")')
        report['checks'].append('In-app manual manifest and lazy chapter fetch through the real desktop scheme')
        snapshot('getting-started','manual-reader-light','图 1：应用内使用说明。左侧选择章节与小节，右侧阅读正文；当前为浅色主题。','public')
        module('M08');click('帮助');until('document.querySelector(".manual-chapter-header h1")?.textContent.includes("变速变调")')
        click('返回 变速变调');until('document.querySelector("#tab-M08")?.getAttribute("aria-selected")==="true"')
        click('帮助');until('document.querySelector(".manual-chapter-header h1")?.textContent.includes("变速变调")')
        assert not js('!![...document.querySelectorAll("dialog[open]")].find(e=>e.innerText.includes("变速变调帮助"))')
        report['checks'].append('Universal module entry and existing help button route to the right chapter and return preserves module')
        if not args.integration_only and not args.details_only and not args.overview_only:
            # Core workflows use the approved natural material, copied to this run.
            choice['folder']=inputs['single'];module('M01');click('打开音频目录')
            until('[...document.querySelectorAll(".m01-file-list button")].some(b=>b.textContent.includes("single-syllable.wav"))',25)
            print('M01 files '+str(js('[...document.querySelectorAll(".m01-file-list button")].map(b=>b.textContent.trim())')),flush=True)
            js('[...document.querySelectorAll(".m01-file-list button")].find(b=>b.textContent.includes("single-syllable.wav")).click()')
            until('!!document.querySelector(".m01-page .wave-track svg")',80)
            report['prepared'].append('M01 approved short natural recording loaded with its supplied sidecar')
            if args.rich or args.flow_only or args.m02_plot_only:
                click('选择输出参数');click('全不选')
                for key in ('pF0','pF1','pF2'):
                    js('document.querySelector('+json.dumps('.parameter-grid input[value="'+key+'"]')+').click()');pause(150)
                until('document.querySelector(".parameter-toolbar .mono")?.textContent.includes("3 / 80")');click('应用到草稿');js('document.querySelector(".m01-file-entry input[type=checkbox]").click()');pause(150);click('分析选中文件')
                until('[...document.querySelectorAll("button")].some(b=>b.offsetParent&&b.textContent.trim()==="保存已完成结果"&&!b.disabled)',120);click('保存已完成结果');until('document.querySelector(".m01-page")?.textContent.includes("已保存")',100)
                snapshot('m01','m01-result-flow-r3-light','参数估计：三项声学参数的真实单音节分析已完成，并保存到授权示例副本目录。')
            module('M02');choice['folder']=inputs['single'];click('打开音频目录',False);click('选择WAV目录',False)
            if args.rich or args.flow_only or args.m02_plot_only:
                until('document.querySelectorAll(".m02-files button").length===2');js('document.querySelector(".m02-files button").click()');until('document.querySelectorAll(".m02-parameters input").length>=3',100)
                click('新建图窗')
                for index in range(3):
                    js('document.querySelectorAll(".m02-parameters input")['+str(index)+'].click()');pause(150)
                click('将 3 项分配到图窗');until('document.querySelectorAll(".parameter-figure .parameter-curve").length===3',60)
                if args.m02_plot_only:
                    assert js('(()=>{const e=[...document.querySelectorAll(".parameter-figure")].find(x=>x.querySelectorAll(".parameter-curve").length===3);const b=[...e.querySelectorAll("button")].find(x=>x.textContent.trim()==="放大图窗");if(!b)return false;b.click();return true})()')
                    until('document.querySelector(".m02-maximized .parameter-chart")?.getBoundingClientRect().height>200',20)
                    pause(600)
                    snapshot('m02','m02-plot-detail-light','参数显示：已将真实音频分析所得三项参数分配到同一图窗，曲线、坐标轴和图例可见。')
                    report['checks'].append('M02 three nonempty parameter curves in a maximized-window screenshot')
                    assert not report['terminations'];report['success']=True
                    return
                snapshot('m02','m02-plots-flow-r3-light','参数显示：读取刚刚保存的同源参数表，将三条参数曲线分配到同一图窗。')
            report['prepared'].append('M02 supplied single-syllable result directory selected when current entry is available')
            module('M03');choice['folder']=inputs['egg'];click('打开音频目录');pause(500)
            js('(()=>{const e=document.querySelector(".egg-page select");if(e?.options.length>1){e.value=e.options[1].value;e.dispatchEvent(new Event("change",{bubbles:true}));}})()')
            until('!!document.querySelector(".egg-page .egg-overview .wave-track svg")',80)
            until('document.querySelector(".egg-live-status")?.textContent.includes("实时预览")',100)
            pause(400);report['prepared'].append('M03 approved stereo EGG input selected; real preview completed')
            if args.flow_only:snapshot('m03','m03-loaded-flow-light','EGG 信号分析：授权声学与 EGG 同步录音已加载，预览和声道选择可见。')
            for mid in ['M04','M06','M07','M08']:
                module(mid);choice['folder']=inputs['single'];click('打开音频目录',False)
                if mid=='M06':select('音频文件');until('!!document.querySelector(".m06-page .wave-track svg")');click('提取参数');until('document.querySelector(".m06-page")?.getAttribute("aria-busy")==="false"',80)
                elif mid=='M07':select('源音频');select('目标音频',2);click('提取 F0');until('document.querySelector(".m07-page")?.getAttribute("aria-busy")==="false"',80)
                elif mid=='M08':
                    js('(()=>{const e=document.querySelector(".m08-page select");if(e?.options.length>1){e.value=e.options[1].value;e.dispatchEvent(new Event("change",{bubbles:true}));}})()')
                    until('!!document.querySelector("svg.m08-curve")');until('document.querySelector(".m08-page")?.getAttribute("aria-busy")==="false"',80);click('合成当前视野');until('document.querySelectorAll(".m08-page .history li").length>0',80)
                elif mid=='M04':
                    select('LPC 音频文件');until('!!document.querySelector(".lpc-page .wave-track svg")');fill('起点','0.1');fill('终点','0.4');click('开始分析');until('!!document.querySelector(".lpc-spectrum svg")',100)
                report['prepared'].append(mid+' natural single-syllable files selected')
                print('Prepared '+mid,flush=True)
                if args.flow_only and mid=='M04':snapshot('m04','m04-analysis-flow-light','LPC 谱图：授权单音节录音选定 0.1–0.4 秒片段并完成真实 LPC 分析。')
                if args.flow_only and mid=='M06':snapshot('m06','m06-extract-flow-light','声学参数合成：从授权短 WAV 提取后，当前参数与轨迹可供核对。')
                if (args.rich or args.flow_only) and mid=='M06':
                    click('合成音频');until('document.querySelector(".m06-page")?.getAttribute("aria-busy")==="false"',100);snapshot('m06','m06-klatt-flow-r3-light','声学参数合成：先提取授权自然录音参数，再显式进行 Klatt 合成；自然录音与合成结果分别显示。')
                if (args.rich or args.flow_only) and mid=='M07':
                    fill('最高 F0 (Hz)',600);click('提取 F0');until('document.querySelector(".m07-page")?.getAttribute("aria-busy")==="false"',100);click('生成当前');until('document.querySelectorAll(".result-audios button").length===10',100);snapshot('m07','m07-nine-flow-r3-light','发声类型合成：以授权源和目标自然录音生成仅 F0 变化的源到目标九步连续统。最高 F0 为600Hz，图中结果 F0 为生成控制轨迹。')
            module('M09');click('音频藏信息');choice['folder']=inputs['single'];click('打开音频目录');select('藏信息源音频');pause(1000)
            if args.rich or args.flow_only:
                click('载入频谱',False);until('document.querySelector(".spectral-editor canvas")?.width>0',80);snapshot('m09','m09-canvas-flow-r3-light','语谱图转音频：授权单音节录音的频谱已载入绘图画布。此模式保留原始相位，黑白画笔调整幅值。')
            module('M12');choice['folder']=inputs['sentence'];click('打开音频目录');until('document.querySelectorAll(".annotation-file-list button").length>0');js('document.querySelector(".annotation-file-list button").click()');until('!!document.querySelector(".annotation-page .wave-track svg")',80)
            if args.flow_only:snapshot('m12','m12-loaded-flow-light','TextGrid 标注：授权自然语句录音与同名 TextGrid 已载入，波形、标注层和保存控件可见。')
            module('M13');pause(700);fill('待转换汉字文本','实验语音学\n重行长乐')
            until('document.querySelector("textarea[aria-label=待转换汉字文本]")?.value.includes("实验语音学")&&Number.parseInt(document.querySelector(".m13-count")?.textContent||"0")>0',25)
            if args.flow_only:snapshot('m13','m13-convert-flow-light','汉字转国际音标：输入示例汉字后，按当前方案显示实时转换结果。')
            module('M17');click('a',False);pause(200)
            if args.flow_only:snapshot('m17','m17-input-flow-light','国际音标表 Plus：点击基础音标后，符号进入可编辑输入区。')
            module('M15');upload('[aria-label="导入刺激 X"]',inputs['single']/'single-syllable.wav');until('document.querySelectorAll(".m15-asset").length>0');report['prepared'].append('M15 genuine authorized single-syllable stimulus imported; no participant session or hearing claim')
            if args.flow_only:snapshot('m15','m15-import-flow-light','感知实验：授权单音节 WAV 已作为刺激 X 导入；画面为设计阶段，没有被试运行。')
            if args.flow_only:
                module('M08');snapshot('m08','m08-result-flow-light','变速变调：授权短音节已生成当前视野的处理结果，原波形、编辑曲线、合成波形与历史对比同屏。')
                module('M07');snapshot('m07','m07-result-flow-light','发声类型合成：源目标 F0 提取和连续统输出完成，可对照轨迹与结果试听。')
                module('M06');snapshot('m06','m06-result-flow-light','声学参数合成：自然原音与 Klatt 合成结果在同一工作台中对照。')
            if not args.flow_only:
                for theme,label in [('light','浅色'),('dark','深色')]:
                    click('设置');click(label);until('document.documentElement.dataset.theme==='+json.dumps(theme))
                    snapshot('settings','settings-'+theme,'图 2：设置中的配色与字体。'+label+'模式通过共同主题即时应用。','public')
                    for mid,title in titles.items():
                        module(mid);snapshot(mid.lower(),mid.lower()+'-overview-'+theme,'图 '+mid[1:]+'-1：'+title+'工作台总览。窗口已最大化；图中控件位置以本版实际界面为准。')
                module('M01');click('选择输出参数');snapshot('m01','m01-parameters','图 01-2：输出参数对话框。按名称或键搜索，勾选后应用到草稿。');click('取消')
                click('编辑14项设置');snapshot('m01','m01-settings','图 01-3：常用分析设置。数值及单位对应下一次分析。');click('REAPER设置 · 4');snapshot('m01','m01-reaper','图 01-4：REAPER设置。新草稿F0范围为30–800Hz，旧草稿保留已存值。');click('取消')
        if args.overview_only:
            for theme,label in [('light','浅色'),('dark','深色')]:
                click('设置');click(label);until('document.documentElement.dataset.theme==='+json.dumps(theme))
                snapshot('settings','settings-'+theme,'设置中的配色与字体。'+label+'模式通过共同主题即时应用。','public')
                for mid,title in titles.items():
                    module(mid);snapshot(mid.lower(),mid.lower()+'-overview-'+theme,title+'工作台总览。窗口已最大化；图中控件位置以本版实际界面为准。')
            module('M01');click('选择输出参数');snapshot('m01','m01-parameters','输出参数对话框。按名称或键搜索，勾选后应用到草稿。');click('取消')
            click('编辑14项设置');snapshot('m01','m01-settings','常用分析设置。数值及单位对应下一次分析。');click('REAPER设置 · 4');snapshot('m01','m01-reaper','REAPER设置。新草稿F0范围为30–800Hz，旧草稿保留已存值。');click('取消')
        if args.details_only:
            module('M13');fill('待转换汉字文本','实验语音学\n重行长乐')
            until('!!document.querySelector(".m13-ambiguous")');js('document.querySelector(".m13-ambiguous").click()');until('!!document.querySelector(".m13-variants")')
            snapshot('m13','m13-polyphony-detail-r4-light','汉字转国际音标：点击带三角标记的多音字，为当前位置选择读音；其他同字位置分别处理。')
            js('document.querySelector("button[aria-label=关闭读音选择]").click()')
            module('M11');click('组件安装与环境检查');until('document.body.innerText.includes("离线组件 ZIP")')
            snapshot('m11','m11-component-detail-r4-light','MFA 自动标注：模型与词典在顶部选择；组件管理区提供现有环境和离线组件的检查入口。')
            module('M14');upload('.phonology-page input[type=file]',ROOT/'tests/fixtures/m14/public.xlsx')
            until('!!document.querySelector("table[aria-label=原始表格样例]")',80)
            snapshot('m14','m14-import-detail-r4-light','音系归纳：公开示例调查表的原始样例，先核对开始行与字头、IPA、备注列。')
            click('确认导入并继续');until('!!document.querySelector("table[aria-label=调类编辑表]")',80)
            snapshot('m14','m14-tone-detail-r4-light','音系归纳：调类编辑页按原调值设置调类名称和顺序，例字与数量来自已导入记录。')
            click('保存调类设置并继续');until('!!document.querySelector(".symbol-editor")')
            snapshot('m14','m14-symbol-detail-r4-light','音系归纳：声韵页内编辑，列表支持选择、排序与归并；保存后才进入结果预览。')
            click('保存声韵设置并继续');click('生成三份结果');until('document.querySelectorAll(".result-files li").length===3',100)
            snapshot('m14','m14-result-detail-r4-light','音系归纳：当前设置生成两份 DOCX 与一份二维 XLSX，原字音和备注保留。')
            module('M15');upload('[aria-label="导入刺激 X"]',inputs['single']/'single-syllable.wav');until('document.querySelectorAll(".m15-asset").length>0');click('参数')
            snapshot('m15','m15-parameters-detail-r4-light','感知实验：时序、随机化、重听、推进方式和按键设置；截图仅展示设计参数，没有进行被试实验或听力确认。')
            from m16_test_backend import Backend
            folder=out/'recording-project';folder.mkdir();choice['folder']=folder
            bridge=w.bridge.recording_bridge();bridge.service.backend=Backend();bridge.choose=lambda purpose:bridge.service.grant(folder,purpose)
            module('M16');until('[...document.querySelectorAll(".recording-page button")].some(b=>b.textContent.trim()==="新建工程"&&!b.disabled)');click('新建工程');until('[...document.querySelectorAll("button")].some(b=>b.textContent.includes("＋ 新任务")&&!b.disabled)')
            click('导入任务');until('!!document.querySelector("[aria-label=录音任务表格格式]")')
            snapshot('m16','m16-import-detail-r4-light','录音：任务表格的列名、次数与文件名说明。工程建在本次专用目录，没有开启麦克风。');click('关闭')
            click('＋ 新任务');until('!!document.querySelector("[aria-label=编辑录音任务]")')
            snapshot('m16','m16-task-detail-r4-light','录音：单条任务编辑，包括编号、名称、朗读内容、导出文件名、分组与备注。');click('完成编辑');click('保存清单')
            module('M17');click('extIPA');snapshot('m17','m17-extipa-detail-r4-light','国际音标表Plus：extIPA 扩展音标，点击输入与范围工具沿用当前光标或选区。')
            click('VoQS');snapshot('m17','m17-voqs-detail-r4-light','国际音标表Plus：VoQS 音质符号，名称与介绍保留正式译表和方法来源。')
            report['checks'].append('Detailed M11–M17 design/import/editor states; no microphone, hearing confirmation or participant session')
        click('设置');click('深色');until('document.documentElement.dataset.theme==="dark"')
        click('使用说明');until('!!document.querySelector(".manual-document")')
        snapshot('getting-started','manual-reader-dark','图 3：深色主题下的使用说明。目录、表格和正文使用同一主题色。','public')
        assert not report['terminations'];report['success']=True
    finally:
        (captures/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        w.closing=True;w.close();pause(300);app.quit()
        print(json.dumps({'success':report['success'],'report':str(captures/'report.json'),'captures':len(report['captures'])},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
