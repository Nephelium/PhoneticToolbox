"""Real localized M08 native gestures, synthesis/history/save and original batch variants."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from verification_artifacts import verification_run
from manual.capture_a_support import Capture

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(args):
 with verification_run('m08-manual',args.locale,'Actual maximized Qt, registered natural syllable, native Shift/mouse gestures, original synthesis and saved WAV readback; two .3-amplitude 1s/16kHz/150+220Hz PCM16 tones for same batch recipe; muted')as(out,scratch):
    from PyQt6.QtCore import QPoint,Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QFileDialog
    c=Capture(args,out,scratch,'m08','pitch-manipulation');print(out,flush=True)
    w=c.window;js,until,click,pause=c.js,c.until,c.click,c.pause;report=c.report
    inputs=scratch/'inputs';inputs.mkdir();saved=scratch/'saved';saved.mkdir()
    source=ROOT/'manual/assets/software-only/single-original.wav';fixture=inputs/'single-syllable.wav';shutil.copyfile(source,fixture)
    report['source']={'path':str(source),'sha256':sha(source)}
    choices={'directory':inputs};QFileDialog.getExistingDirectory=lambda *a,**k:str(choices['directory'])
    QFileDialog.getSaveFileName=lambda *a,**k:(str(saved/Path(a[2]).name),'')
    def fill(label,value):
        import re
        match=re.fullmatch(r'(时间|频率|连接) (\d+)',label)
        if match:
            index=int(match[2]);column={'时间':2,'频率':3,'连接':4}[match[1]]
            return c.fill(f'dialog[open] tbody tr:nth-child({index+1}) td:nth-child({column}) '+('select'if column==4 else'input'),value)
        c.fill('[aria-label='+json.dumps(c.ui.label(label),ensure_ascii=False)+']',value)
    def idle():until('document.querySelector(".m08-page")?.getAttribute("aria-busy")==="false"',180)
    def dismiss_save():
        if js('!!document.querySelector("dialog[open] .file-list")'):click('取消')
    def snapshot(name,caption,state):
        if args.scene and name not in args.scene:return
        regions={
          'm08-freehand-reference-light':{'curve':'svg.m08-curve'},
          'm08-independent-selection-light':{'wave':'.plots>.module-section:nth-child(3)','selection':'.plots>.module-section:nth-child(3) .wave-selection'},
          'm08-f0-import-light':{'dialog':'dialog[open]','values':'dialog[open] textarea'},
          'm08-breakpoints-three-light':{'dialog':'dialog[open]','row':'dialog[open] tbody tr:nth-child(2)'},
          'm08-history-audition-light':{'history':'.history-details','audition':'.history li:nth-child(3)>button'}
        }.get(name,{})
        c.shot(name,regions);report['captures'][-1].update(caption=caption,state=state)
    def capture_variants():
        import numpy as np
        from scipy.io import wavfile
        tones=scratch/'tones';tones.mkdir()
        for name,hz in [('原音 ɑ̃˥',150),('原音 二',220)]:
            t=np.arange(16000)/16000;wavfile.write(tones/(name+'.wav'),16000,(.3*32767*np.sin(2*np.pi*hz*t)).astype('int16'))
        report['batchInputs']={p.name:sha(p)for p in tones.iterdir()}
        choices['directory']=tones;click('打开音频目录');click('批量变速变调')
        fill('音高调整方式','hz');fill('语速倍率','0.8, 1.2');fill('音高偏移 Hz','-20, 20');click('全选音频')
        until('document.querySelectorAll(".m08-page .file-list input:checked").length===2')
        snapshot('m08-multi-variants-20261021-6e6f6396','Original two synthetic tones, two speeds and two Hz offsets: four versions/file, eight total.',{'sources':2,'speed':[.8,1.2],'pitch_hz':[-20,20],'versions':8})
    def drag(selector, points, shift=False):
        box = js('(()=>{const r=document.querySelector(' + json.dumps(selector) + ').getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()')
        target = w.view.focusProxy() or w.view
        coords = [QPoint(round(box['x'] + x * box['w']), round(box['y'] + y * box['h'])) for x, y in points]
        modifiers = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
        if shift:
            QTest.keyPress(target, Qt.Key.Key_Shift)
        QTest.mousePress(target, Qt.MouseButton.LeftButton, modifiers, coords[0])
        for point in coords[1:]:
            QTest.mouseMove(target, point, 30)
            pause(40)
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, modifiers, coords[-1])
        if shift:
            QTest.keyRelease(target, Qt.Key.Key_Shift)
        pause(250)

    try:
        until('!!document.querySelector(".m08-page")');c.prepare()
        click('打开音频目录')
        until('document.querySelector(".m08-page select")?.options.length===2')
        js('(()=>{const e=document.querySelector(".m08-page select");e.value=e.options[1].value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('!!document.querySelector("svg.m08-curve")')
        idle()
        assert js('document.querySelectorAll(".history li").length') == 0
        snapshot('m08-loaded-empty-light', '短音节载入后，原音波形和原始 F0 已出现；尚未生成音频，输出与历史图为空。', {'historyCount': 0})
        fill('参考线 Hz', 200)
        click('添加参考线')
        before = js('document.querySelector("svg.m08-curve .modified").getAttribute("d")')
        drag('svg.m08-curve', [(0.23, .63), (.30, .55), (.40, .42), (.49, .32), (.58, .38), (.67, .51), (.77, .60)], shift=True)
        after = js('document.querySelector("svg.m08-curve .modified").getAttribute("d")')
        assert before != after, 'Native Shift drag must change the actual target path'
        snapshot('m08-freehand-reference-light', '按 Shift 拖动后的红色目标曲线与原始虚线分开，200 Hz 参考线保持可见；此时尚未合成。', {'historyCount': 0, 'nativeShiftDrag': True, 'referenceHz': 200})
        click('合成整段')
        until('document.querySelectorAll(".history li").length===1')
        idle()
        dismiss_save()
        snapshot('m08-freehand-result-light', '手绘目标完成整段合成，输出波形和从实际 WAV 重新估计的历史 F0 已出现。', {'historyCount': 1, 'action': 'synthesize', 'speed': 1})
        drag('.plots>.module-section:nth-child(1) .wave-track svg', [(.20, .50), (.47, .50)])
        drag('.plots>.module-section:nth-child(3) .wave-track svg', [(.35, .50), (.70, .50)])
        selection_widths = js('[...document.querySelectorAll(".plots .wave-selection")].map(e=>Number(e.getAttribute("width")))')
        assert selection_widths[0] == 0 and selection_widths[1] > 0, selection_widths
        snapshot('m08-independent-selection-light', '合成波形已拖出试听选区，右栏显示合成音选区；切换到这份音频时原音试听选区清除，原音计算视野仍为整段。', {'historyCount': 1, 'playbackSelectionIndependentOfView': True, 'waveSelectionWidths': selection_widths})
        click('导入基频序列')
        fill('基频序列', '140\n180\n220\n180\n140')
        snapshot('m08-f0-import-light', '基频序列窗口中每行填写一个 Hz 值；导入会按当前视野内唯一连续有声段均匀重采样。', {'dialog': '导入基频序列', 'valuesHz': [140, 180, 220, 180, 140], 'applied': False})
        click('取消')
        click('编辑拐点表')
        click('添加行')
        for i, t, f, mode in [(0, .10, '140', 'constant'), (1, .32, '180,220,260', 'order'), (2, .57, '150', 'constant')]:
            fill('时间 ' + str(i), t)
            fill('频率 ' + str(i), f)
            fill('连接 ' + str(i), mode)
        snapshot('m08-breakpoints-three-light', '起点 140 Hz、终点 150 Hz 设为常量，中间 0.32 s 填入三个顺序值，准备生成三个不同峰值的版本。', {'dialog': '编辑拐点表', 'points': [{'time': .10, 'freqs': [140], 'mode': 'constant'}, {'time': .32, 'freqs': [180, 220, 260], 'mode': 'order'}, {'time': .57, 'freqs': [150], 'mode': 'constant'}]})
        click('保存更改')
        until('!document.querySelector("dialog[open]")')
        click('批量生成')
        until('document.querySelectorAll(".history li").length===4')
        idle()
        dismiss_save()
        snapshot('m08-three-path-history-light', '拐点工具已生成三个峰值不同的实际音频，历史图同时显示先前手绘版本及三个拐点版本。', {'historyCount': 4, 'linearResultCount': 3})
        js('document.querySelectorAll(".history li")[2].querySelector("button").click()')
        pause(250)
        js('document.querySelectorAll(".history li")[2].querySelector("details").open=true')
        js('(()=>{const row=document.querySelectorAll(".history li")[2],list=document.querySelector(".history"),body=document.querySelector(".workbench-right-body"),section=document.querySelector(".history-details");list.scrollTop+=row.getBoundingClientRect().top-list.getBoundingClientRect().top;body.scrollTop+=section.getBoundingClientRect().top-body.getBoundingClientRect().top;})()')
        snapshot('m08-history-audition-light', '点击历史中的试听此版本后，合成波形切到所选版本；展开生成参数可核对这一次提交的拐点与范围。', {'historyCount': 4, 'selectedHistoryIndex': 2, 'parametersExpanded': True, 'mutedPhysicalOutput': True})
        js('document.querySelectorAll(".history li")[2].querySelector("details").open=false;document.querySelector(".workbench-right-body").scrollTop=0')
        click('批量保存')
        snapshot('m08-save-selection-light', '批量保存窗口列出当前视野的四份已生成 WAV，默认全部勾选；此步骤尚未选择外部保存目录。', {'dialog': '批量保存', 'selectedCount': 4, 'saved': False})
        choices['directory']=saved
        click('保存所选音频')
        until('!document.querySelector("dialog[open]")')
        idle()
        until('!!document.querySelector(".save-location")')
        js('document.querySelector(".save-location details").open=true')
        assert len(list(saved.glob('*.wav'))) == 4
        snapshot('m08-save-confirmed-light', '四个音频已保存，最近保存提示给出实际目录和最终文件名；生成历史仍保留。', {'historyCount': 4, 'savedCount': 4, 'directory': str(saved)})
        report['saved'] = [{'name': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(saved.glob('*.wav'))]
        capture_variants()
        assert sha(source)==sha(fixture)==report['source']['sha256']
        report['checks'] += ['Actual natural-syllable original, native Shift drag, three generated breakpoint outputs and four saved WAVs', 'Two original synthetic-tone fixtures, speed .8/1.2, exclusive Hz offsets -20/20, eight planned variants', 'Original input SHA unchanged; temporary WAVs cleaned with owned verification scratch']
        report['success']=True
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:c.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--locale',choices=('en','ja'),required=True);p.add_argument('--scene',action='append',default=[]);p.add_argument('--frontend-dist',type=Path,default=ROOT/'frontend/dist');run(p.parse_args())
