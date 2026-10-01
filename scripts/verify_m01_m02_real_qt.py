"""Actual Qt regression using only user-provided recordings, never synthesized audio."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4
import soundfile as sf
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtWidgets import QApplication, QFileDialog
from ptb_desktop.host import Workbench, register_scheme
from ptb_desktop import annotation_audio

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--skip-batch', action='store_true')
    args = parser.parse_args()
    out = ROOT/'output/validation/m01-m02-r1'/('real-qt-'+uuid4().hex)
    exports = out/'segments';exports.mkdir(parents=True)
    originals = {p:digest(p) for p in args.corpus.iterdir() if p.is_file()}
    count = len(list(args.corpus.glob('*.wav')))
    config = json.loads((ROOT/'output/validation/m01/workbench-local.json').read_text('utf-8'))
    register_scheme();app = QApplication(['M01 M02 real recording verification'])
    window = Workbench(ROOT/'frontend/dist', test=True,
        jobs_path=None if args.skip_batch else ROOT/'output/validation/p06/local-state.sqlite3',
        local_files_root=None if args.skip_batch else ROOT/'output/validation/m01'/config['cache'],
        vocal_profile=out/'vocal')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(1600,1000);window.show()
    folder = [args.corpus]
    QFileDialog.getExistingDirectory = lambda *a,**k:str(folder[0])
    saved = []
    def save_dialog(*a,**k):
        name=Path(a[2]).name;saved.append(name);return str(out/name),''
    QFileDialog.getSaveFileName=save_dialog
    report={'success':False,'synthetic_audio':False,'checks':[],'long_audio':[],'schema_applied':[]}
    original_preview=annotation_audio.preview_audio
    service_preview=window.service.preview
    def observed_preview(payload,query):
        record={'query':query,'bytes':len(payload)}
        report.setdefault('spectrogram_requests',[]).append(record)
        try:
            result=service_preview(payload,query);record['success']=True;return result
        except Exception as exc:
            record['error']=str(exc);raise
    window.service.preview=observed_preview
    def js(code):
        loop=QEventLoop();values=[]
        window.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()))
        QTimer.singleShot(6000,loop.quit);loop.exec();return values[0] if values else None
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def until(code,seconds=90):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' / '+str(js('document.querySelector(".error-banner,[role=alert]")?.textContent')))
    def click(text):
        code='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&!b.disabled&&b.textContent.trim()==='+json.dumps(text)+')'
        until('!!'+code);js(code+'?.click()');pause()
    def checked(label,value):
        code='[...document.querySelectorAll("label")].find(e=>e.offsetParent&&e.textContent.includes('+json.dumps(label)+'))?.querySelector("input[type=checkbox]")'
        js('(()=>{const e='+code+';if(e&&e.checked!=='+json.dumps(value)+')e.click();})()');pause()
    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        click('参数估计');click('选择音频目录')
        until(f'document.querySelectorAll(".m01-file-entry").length==={count}')
        js('document.querySelector(".m01-file-list .file-row").click()')
        until('document.querySelectorAll(".m01-tiers option").length===2&&!!document.querySelector(".wave-track")')
        for height in (1000,660):
            window.resize(1600,height);pause(600)
            assert js('(()=>{const p=document.querySelector(".workbench-columns>.file-panel");if(!p)return false;p.scrollTop=500;return p.scrollTop>0&&p.clientHeight<window.innerHeight;})()')
        window.resize(1600,1000);pause(600);window.view.grab().save(str(out/'m01-real.png'))
        report['checks'].append(f'{count} real WAV/TextGrid pairs; bounded list at 1000/660px')
        if not args.skip_batch:
            checked('结果与WAV同目录',False);folder[0]=exports;click('选择结果目录')
            js('document.querySelector("input[aria-label=全选音频]").click()')
            click('保存当前层切分音频')
            until('document.querySelector(".task-operation-status")?.style.visibility==="hidden"')
            observations=[]
            for _ in range(80):
                observations.append(js('(()=>{const b=[...document.querySelectorAll("button")].find(e=>e.textContent.trim()==="开始全列表分析");return [b.getBoundingClientRect().y,b.disabled,document.querySelector(".task-operation-status").textContent];})()'))
                pause(100)
            assert all(abs(row[0]-observations[0][0])<.1 and not row[1] and '正在' not in row[2] for row in observations)
            print('Real batch submitted; waiting for segmentation and native saves',flush=True)
            until(f'document.querySelector(".m01-results strong")?.textContent.includes("{count} / {count}")',240)
            until('document.querySelector(".m01-page")?.textContent.includes("已保存")',120)
            assert list(exports.glob('*.wav'))
            report['batch_export_files']=len(list(exports.iterdir()))
            report['checks'].append(f'Actual {count}-file TextGrid batch succeeds, saves only to validation directory; start button remains stable across 80 observations')
            pause(600);window.view.grab().save(str(out/'m01-real-batch.png'))
        click('参数显示');folder[0]=args.corpus;click('选择音频目录')
        until(f'document.querySelectorAll(".m02-files button").length==={count}')
        js('document.querySelector(".m02-files button").click()')
        until('document.querySelectorAll(".m02-parameters input").length>10')
        js('document.querySelector(".m02-parameters input").click()');click('将 1 项分配到图窗')
        until('!!document.querySelector(".parameter-curve")');click('保存当前图')
        end=time.monotonic()+20
        while not saved or not (out/saved[0]).exists():
            assert time.monotonic()<end;pause()
        assert saved[0].endswith('.png') and (out/saved[0]).read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
        pause(600);window.view.grab().save(str(out/'m02-real.png'))
        click('清空选定图窗');until('!document.querySelector(".parameter-curve")')
        click('删除选定图窗');until('!document.querySelector(".parameter-figure")')
        click('新建图窗');until('!!document.querySelector(".empty-plot")')
        report['checks'].append('Actual parameter table; default native PNG signature; clear/delete last/create plot')
        # Exact copies of the authorized recording exercise nested grants. Never
        # scan sibling corpora or build synthetic/concatenated audio.
        recursive=out/'recursive';recursive.mkdir()
        source=next(args.corpus.glob('*.wav'))
        for sub in ('甲','乙'):
            target=recursive/sub;target.mkdir()
            for path in (source,source.with_suffix('.TextGrid')):
                (target/path.name).write_bytes(path.read_bytes())
                assert digest(target/path.name)==originals[path]
        folder[0]=recursive;click('选择音频目录')
        checked('包含子文件夹',True)
        until('document.querySelectorAll(".m02-files button").length===2')
        report['recursive_real_audio_count']=js('document.querySelectorAll(".m02-files button").length')
        pause(600);window.view.grab().save(str(out/'m02-real-recursive.png'))
        checked('包含子文件夹',False)
        # Exercise the same compact-preview path with unmodified real audio and
        # a lower test-only byte budget. This does not verify a real long file.
        annotation_audio.preview_audio=lambda stream,max_bytes,**kwargs:original_preview(stream,min(max_bytes,160_000),**kwargs)
        window.provider.preview_cache=None
        for index,path in enumerate([source]):
            info=sf.info(path);duration=info.frames/info.samplerate
            for module,selector in [('参数显示','.m02-page'),('参数估计','.m01-page')]:
                click(module);folder[0]=path.parent;click('选择音频目录')
                start=time.monotonic()
                if module=='参数估计':
                    until('!document.querySelector(".m01-page .audio-loading")')
                    js('[...document.querySelectorAll(".m01-file-list .file-row")].find(b=>b.textContent.trim().startsWith('+json.dumps(path.name)+'))?.click()')
                else:click(path.name)
                until(f'!!document.querySelector("{selector} .wave-track")&&!document.querySelector("{selector} .audio-loading")')
                expected=duration.__format__('.3f')
                until(f'document.querySelector("{selector}")?.textContent.includes('+json.dumps(expected)+')')
                checked('显示两个声道',info.channels>1)
                if module=='参数显示':
                    js('(()=>{const e=document.querySelectorAll(".m02-toolbar input[type=number]")[1];e.value="10";e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
                    js('(()=>{const e=document.querySelector(".m02-toolbar input[type=number]");e.value='+str(max(0,duration-10))+';e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
                checked('显示语谱图',True)
                until(f'!!document.querySelector("{selector} .spectrogram-canvas canvas")&&!document.querySelector("{selector} .spectrogram-view [role=status]")')
                assert not js(f'document.querySelector("{selector} .spectrogram-view [role=alert]")?.textContent')
                _,(raw,sha,source_duration,note)=window.provider.preview_cache
                assert sha==originals[path] and source_duration==duration and len(raw)<=64_000_000
                assert sf.info(__import__('io').BytesIO(raw)).channels==info.channels
                report.setdefault('real_compact_preview',[]).append(dict(name=path.name,module=module,test_byte_budget=160_000,bytes=path.stat().st_size,duration=duration,seconds=time.monotonic()-start,preview_bytes=len(raw),note=note))
                js(f'document.querySelector("{selector} .signal-panel").scrollTop=500');pause(600)
                window.view.grab().save(str(out/f'real-compact-{index}-{module}.png'))
                print(f'Actual audio compact path passed: {path.name} / {module}',flush=True)
        annotation_audio.preview_audio=original_preview
        assert all(digest(path)==sha for path,sha in originals.items())
        report['source_hashes_unchanged']=True;report['success']=True
    except Exception as exc:
        report['error']=str(exc);window.view.grab().save(str(out/'failed.png'));raise
    finally:
        annotation_audio.preview_audio=original_preview
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
        print(out,flush=True);window.closing=True;window.close();app.processEvents()


if __name__=='__main__':main()
