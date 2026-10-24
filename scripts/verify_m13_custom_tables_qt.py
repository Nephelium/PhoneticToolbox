"""Current-source native Qt import/save/PNG, with an off-record owned profile."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verification_artifacts import verification_run
from verify_m13_custom_tables import fixtures
from workbench_source import bind_sources

ROOT = Path(__file__).resolve().parents[1]
bind_sources(ROOT)
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')


def run(out, scratch):
    from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from PyQt6.QtGui import QColor, QImage, QPainter
    from ptb_desktop.host import Workbench, Page, register_scheme
    register_scheme()
    app = QApplication(['M13-custom-tables-QA'])
    chosen = [str(scratch / 'hanzi.csv')]
    Page.chooseFiles = lambda self, mode, old, accepted: chosen
    w = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=None, vocal_profile=scratch / 'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    w.resize(1440, 900)
    w.show()
    report = {'success': False, 'checks': [], 'scope': 'actual Windows Qt, off-record profile, native QTest import/save; refresh persistence, no final EXE/macOS'}
    target = scratch / 'custom.png'
    QFileDialog.getSaveFileName = lambda *args, **kwargs: (str(target), 'PNG')

    def pause(ms=80):
        loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()

    def js(code):
        loop, box = QEventLoop(), []
        w.page.runJavaScript(code, lambda value: (box.append(value), loop.quit()))
        QTimer.singleShot(6000, loop.quit); loop.exec()
        assert box, 'JavaScript timeout'
        return box[0]

    def until(code, seconds=30):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if js(code): return
            pause()
        raise AssertionError(code + ' ' + str(js('document.body.innerText.slice(-2500)')))

    def click(text):
        code = '(()=>{const name=' + json.dumps(text) + ';const e=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&(e.textContent.trim()===name||e.getAttribute("aria-label")===name||e.matches(".nav-item")&&e.textContent.includes(name)));if(!e||e.disabled)return false;e.click();return true})()'
        assert js(code), text
        pause()

    def fill(label, value):
        js('(()=>{const e=document.querySelector(' + json.dumps('[aria-label="'+label+'"]') + ');e.value=' + json.dumps(value) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        pause()

    def native(selector):
        point = js('(()=>{const e=document.querySelector(' + json.dumps(selector) + ');e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return {x:r.x+Math.min(50,r.width/2),y:r.y+r.height/2}})()')
        pause(); QTest.mouseClick(w.view.focusProxy() or w.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(round(point['x']), round(point['y']))); pause()

    def values(expected):
        until('[...document.querySelectorAll(".m13-mapped")].map(e=>e.dataset.value).join("|")===' + json.dumps(expected))

    try:
        until('!!document.querySelector("nav")'); click('汉字转国际音标')
        until('!!document.querySelector("[aria-label=转换标准]")&&!document.querySelector("[aria-label=转换标准]").disabled')
        click('导入自定义字表'); native('[aria-label=选择字表文件]')
        until('document.querySelectorAll("[aria-label=导入结果预览] tbody tr").length===3'); fill('字表名称', 'Qt调查表')
        native('dialog .dialog-actions .primary'); until('!document.querySelector("dialog")')
        fill('待转换汉字文本', '行𠮷妈'); values('hɑŋ˧˥|kiʔ|?')
        native('.m13-ambiguous'); click('读音 2ɕiŋ˧˥'); values('ɕiŋ˧˥|kiʔ|?')
        report['checks'].append('Native QTest file chooser, named CSV save, supplementary Hanzi, two readings, missing mapping')
        click('保存本机草稿 *')
        loaded = QEventLoop()
        w.view.loadFinished.connect(loaded.quit)
        w.view.reload(); QTimer.singleShot(30000, loaded.quit); loaded.exec()
        w.view.loadFinished.disconnect(loaded.quit)
        until('!!document.querySelector(".home-page")'); click('汉字转国际音标')
        until('!!document.querySelector("[aria-label=转换标准]")&&!document.querySelector("[aria-label=转换标准]").disabled')
        assert js('document.querySelector("[aria-label=待转换汉字文本]").value===""&&document.querySelector("[aria-label=转换标准] option:checked").textContent==="Qt调查表"')
        click('恢复本机草稿'); values('ɕiŋ˧˥|kiʔ|?')
        report['checks'].append('Actual Qt reload restores library and selection, with separate complete draft restoration')
        chosen[:] = [str(scratch / 'shared.xlsx')]
        click('导入自定义字表'); native('[aria-label=选择字表文件]')
        until('document.querySelectorAll("[aria-label=导入结果预览] tbody tr").length===1'); assert js('document.querySelector("[aria-label=导入结果预览] tbody").textContent.includes("mɑ˥")')
        fill('字表工作表', '1'); until('document.querySelector("[aria-label=导入结果预览] tbody").textContent.includes("nɯ˩")')
        fill('字表名称', 'Qt工作表'); w.view.grab().save(str(out / 'qt-import.png')); native('dialog .dialog-actions .primary'); until('!document.querySelector("dialog")')
        fill('待转换汉字文本', '女'); values('nɯ˩')
        report['checks'].append('Qt Chromium DEFLATE XLSX, shared and inline strings, multiple sheets')
        js('window.__paint=[];const original=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(text,...args){__paint.push(text);return original.call(this,text,...args)}')
        click('保存为 PNG'); until('!!document.querySelector(".png-preview img")&&!document.querySelector("dialog .primary").disabled'); click('导出 PNG'); end = time.monotonic() + 30
        while not target.exists() and time.monotonic() < end: pause()
        assert target.exists() and target.read_bytes()[1:4] == b'PNG'
        assert js('__paint.includes("Qt工作表")&&__paint.includes("nɯ˩")')
        base=js('getComputedStyle(document.documentElement).getPropertyValue("--app").trim()')
        assert QImage(str(target)).pixelColor(2,2).name()==QColor(base).name()
        image=QImage(1200,600,QImage.Format.Format_RGB32);image.fill(QColor('#ffffff'));painter=QPainter(image);painter.fillRect(0,0,600,600,QColor('#000000'));painter.end();source_image=scratch/'split.png';assert image.save(str(source_image))
        chosen[:]=[str(source_image)];click('保存为 PNG')
        until('!!document.querySelector(".png-preview img")&&!document.querySelector("dialog .primary").disabled')
        js('document.querySelector("input[name=png-background][value=image]").click()');pause()
        native('input[aria-label="PNG 背景图片"]');fill('PNG 高斯模糊',12);fill('PNG 图片不透明度',100)
        until('!!document.querySelector(".png-preview img")&&!document.querySelector("dialog .primary").disabled')
        assert js('[...document.querySelectorAll("input")].find(e=>e.getAttribute("aria-label")==="PNG 高斯模糊").value==="12"')
        until('document.querySelector(".png-preview img").complete');pause(180);w.view.grab().save(str(out/'qt-png-background.png'));target=scratch/'background-blurred.png';click('导出 PNG')
        end=time.monotonic()+30
        while not target.exists() and time.monotonic()<end:pause()
        result=QImage(str(target));assert not result.isNull();y=int(result.height()*.92)
        assert result.pixelColor(2,y).red()==0 and result.pixelColor(result.width()-3,y).red()==255
        assert 0<result.pixelColor(result.width()//2-5,y).red()<255
        report['checks'].append('Hidden Qt PNG settings preview, current background colour, local image chooser, Gaussian blur and native saved PNG pixel readback')
        for theme in ['light', 'dark']:
            js('document.documentElement.dataset.theme=' + json.dumps(theme)); pause(250)
            w.view.grab().save(str(out / f'qt-result-{theme}.png'))
        click('保存本机草稿 *')
        original_id = js('document.querySelector("[aria-label=转换标准]").value')
        chosen[:] = [str(scratch / 'replacement.csv')]
        click('替换字表'); native('[aria-label=选择字表文件]')
        until('document.querySelectorAll("[aria-label=导入结果预览] tbody tr").length===2'); fill('字表名称', 'Qt替换版')
        w.view.grab().save(str(out / 'qt-replace.png')); native('dialog .dialog-actions .primary'); until('!document.querySelector("dialog")')
        assert js('document.querySelector("[aria-label=转换标准]").value') == original_id
        fill('待转换汉字文本', '行'); values('a˥')
        click('删除字表'); click('取消'); values('a˥')
        click('删除字表'); w.view.grab().save(str(out / 'qt-delete.png')); native('dialog .dialog-actions .danger'); until('!document.querySelector("dialog")')
        assert js('document.querySelector("[aria-label=待转换汉字文本]").value==="行"&&document.querySelector("[aria-label=转换标准]").value==="Standard Chinese (Beijing)"&&document.querySelectorAll("[aria-label=转换标准] optgroup option").length===1')
        # This reload intentionally discards the current text to inspect the older saved draft.
        waiting_for_discard = [True]
        def confirm_discard():
            box = QApplication.activeModalWidget()
            if isinstance(box, QMessageBox) and box.windowTitle() == '未保存的草稿':
                QTest.mouseClick(box.button(QMessageBox.StandardButton.Yes), Qt.MouseButton.LeftButton)
                report['nativeDiscardConfirmed'] = True
            elif waiting_for_discard[0]:
                QTimer.singleShot(50, confirm_discard)
        QTimer.singleShot(0, confirm_discard)
        loaded = QEventLoop(); w.view.loadFinished.connect(loaded.quit)
        w.view.reload(); QTimer.singleShot(30000, loaded.quit); loaded.exec(); w.view.loadFinished.disconnect(loaded.quit)
        waiting_for_discard[0] = False
        assert report.get('nativeDiscardConfirmed'), 'Expected native unsaved-edit confirmation'
        until('!!document.querySelector(".home-page")'); click('汉字转国际音标')
        until('!!document.querySelector("[aria-label=转换标准]")&&!document.querySelector("[aria-label=转换标准]").disabled')
        click('恢复本机草稿')
        assert js('document.querySelector("[aria-label=待转换汉字文本]").value==="女"&&document.querySelectorAll("[aria-label=转换标准] optgroup option").length===1&&document.body.innerText.includes("已删除或不可用")')
        report['checks'].append('Native replacement retains identity, delete/cancel preserves text and other tables, reload removes table permanently while old draft text recovers')
        chosen[:] = [str(scratch / 'wide.tsv')]
        click('导入自定义字表'); native('[aria-label=选择字表文件]')
        until('document.querySelector("[aria-label=数据起始行]")?.value==="4"')
        fill('汉字或音节列','0'); assert js('document.querySelector("dialog .primary").disabled')
        fill('汉字或音节列','3'); fill('音标列','1'); fill('数据起始行','3'); assert js('document.querySelector("dialog .primary").disabled')
        fill('数据起始行','4'); fill('音标声调表示方式','values'); fill('字表名称','Qt任意列调值')
        until('document.querySelector("[aria-label=导入结果预览]").textContent.includes("ma˨˩˦")')
        w.view.grab().save(str(out / 'qt-columns-tones.png')); native('dialog .dialog-actions .primary'); until('!document.querySelector("dialog")')
        fill('待转换汉字文本','妈麻马'); values('ma˥˥|ma˧˥|ma˨˩˦'); click('显示声调'); values('ma|ma|ma'); click('显示声调')
        report['checks'].append('Actual Qt arbitrary columns/start row with raw preview, extra notes ignored, numeric/superscript/fullwidth tone values and tone toggle')
        chosen[:] = [str(scratch / 'classes.tsv')]
        click('导入自定义字表'); native('[aria-label=选择字表文件]'); until('!!document.querySelector("[aria-label=音标声调表示方式]")')
        fill('字表名称','Qt调类'); fill('音标声调表示方式','classes'); native('dialog .dialog-actions .primary'); until('!document.querySelector("dialog")'); values('ma¹|ma2|ma３')
        click('声调映射'); fill('调类显示方式','values'); assert js('document.querySelector("dialog .primary").disabled')
        fill('调类 1 的调值','33'); fill('调类 2 的调值','21'); fill('调类 3 的调值','214')
        js('document.documentElement.dataset.theme="dark"'); pause(250); w.view.grab().save(str(out / 'qt-tone-mapping-dark.png'))
        native('dialog .dialog-actions .primary'); until('!document.querySelector("dialog")'); values('ma˧˧|ma˨˩|ma˨˩˦'); click('保存本机草稿 *')
        loaded=QEventLoop();w.view.loadFinished.connect(loaded.quit);w.view.reload();QTimer.singleShot(30000,loaded.quit);loaded.exec();w.view.loadFinished.disconnect(loaded.quit)
        until('!!document.querySelector(".home-page")');click('汉字转国际音标');until('!!document.querySelector("[aria-label=转换标准]")&&!document.querySelector("[aria-label=转换标准]").disabled');click('恢复本机草稿');values('ma˧˧|ma˨˩|ma˨˩˦')
        click('声调映射'); assert js('document.querySelectorAll("[aria-label]").length>0&&[...document.querySelectorAll("input")].find(e=>e.getAttribute("aria-label")==="调类 1 的调值").value==="33"')
        fill('调类显示方式','original'); native('dialog .dialog-actions .primary');until('!document.querySelector("dialog")');values('ma¹|ma2|ma３')
        report['checks'].append('Actual Qt editable class mapping persists after reload and restores original numeric labels without reimport')
        click('帮助'); until('document.body.innerText.includes("导入并保存自己的对照表")')
        report['checks'].append('Native PNG file and custom name/IPA, light/dark screenshots, generated manual help')
        report['success'] = True
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc(), body=js('document.body.innerText'))
        w.view.grab().save(str(out / 'failure.png'))
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        w.closing = True; w.close(); pause(400)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        run(Path(os.environ['M13_QA_OUT']), Path(os.environ['M13_QA_SCRATCH']))
    else:
        with verification_run('m13-custom-tables', 'qt', 'Owned Windows Qt; synthetic tables, QTest file import, save and PNG; no user database') as (out, scratch):
            fixtures(scratch)
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--worker'],
                env=dict(os.environ, M13_QA_OUT=str(out), M13_QA_SCRATCH=str(scratch)), capture_output=True, text=True, encoding='utf-8', errors='replace')
            (out / 'console.log').write_text(result.stdout + result.stderr, encoding='utf-8')
            print(out, flush=True)
            if result.returncode:
                print(result.stderr, flush=True)
                raise RuntimeError('M13 custom table Qt acceptance failed')
