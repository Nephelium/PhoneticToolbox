"""Visible desktop source UI, real offline prediction and installed neural tasks.

Uses an owned DB/cache and synthetic image. Does not download, build, install,
or modify existing components or user windows. Runs with the default GPU.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import time

from verification_artifacts import verification_run, write_json
from workbench_source import configure

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--python",type=Path,required=True);args=parser.parse_args()
    configure(ROOT)
    import cv2
    import numpy as np
    import soundfile as sf
    from PyQt6.QtCore import QEventLoop, QTimer, QCoreApplication, QEvent
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    from ptb_worker.vocoder_components import root as component_root, status
    assert status()['pipeline_ready'], 'This integration test requires the existing locally installed component'
    existing_component_root = str(component_root())
    assert '--disable-gpu' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
    register_scheme()
    app = QApplication(['M09-built-in-owned-QA'])
    app.setQuitOnLastWindowClosed(False)
    with verification_run('m09-parameters', 'desktop', 'Visible default-GPU Qt, real WebChannel/API/worker; absent optional component prediction then installed GL/repair/vocoder') as (out, scratch):
        os.environ['PTB_M09_PYTHON_CONFIG']=str(scratch/'gpu-python.json')
        inputs = scratch/'inputs';inputs.mkdir()
        gray = np.full((96, 144), 240, np.uint8)
        for i in range(10, 90, 16):cv2.line(gray, (0, i), (143, i+4), 50, 2)
        for i in range(0, 144, 12):cv2.line(gray, (i, 8), (i, 90), 150, 1)
        (inputs/'synthetic.png').write_bytes(cv2.imencode('.png', gray)[1].tobytes())
        report = dict(success=False, cases=[], flags=os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', ''), scope='Windows source desktop; default GPU; existing component, no download or build')
        w = None

        def pause(ms=100):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()

        def js(code):
            loop=QEventLoop();values=[]
            w.page.runJavaScript(code, lambda v:(values.append(v),loop.quit()))
            QTimer.singleShot(10000,loop.quit);loop.exec()
            assert values, 'JavaScript timeout'
            return values[0]

        def until(code, seconds=90):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if js('Boolean('+code+')'):return
                pause()
            raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-2500)')))

        def click(text):
            assert js("(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&e.textContent.trim()==="+json.dumps(text)+");if(!e||e.disabled)return false;e.click();return true})()"),text
            pause()

        def select(label, value):
            assert js("(()=>{const e=document.querySelector('select[aria-label=\""+label+"\"]');if(!e)return false;e.value="+json.dumps(value)+";e.dispatchEvent(new Event('change',{bubbles:true}));return true})()")
            pause()

        def fill(label, value):
            assert js("(()=>{const e=[...document.querySelectorAll('label')].find(e=>e.textContent.trim().startsWith("+json.dumps(label)+"))?.querySelector('input');if(!e||e.disabled)return false;e.value="+json.dumps(str(value))+";e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));return true})()")
            pause()

        def close_window():
            nonlocal w
            if w is not None:
                js('window.onbeforeunload=null');w.closing=True;w.close();w.service.close();pause(500)
                w.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);pause(150);w=None

        try:
            for installed in (False, True):
                case=scratch/('installed' if installed else 'offline');case.mkdir()
                os.environ['PTB_VOCODER_ROOT']=existing_component_root if installed else str(case/'absent')
                db=case/'jobs.sqlite3'
                with sqlite3.connect(db) as target:
                    for migration in ('002_jobs_sqlite.sql','005_acoustic_batches_sqlite.sql'):
                        target.executescript((ROOT/'backend/migrations'/migration).read_text('utf-8'))
                cache=case/'cache';cache.mkdir();initialize_local_files(cache)
                saved=case/'saved';saved.mkdir()
                QFileDialog.getOpenFileName=lambda *a,**k:(str(args.python),'')
                QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) else inputs)
                w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=case/'vocal',start_module='M09')
                w.resize(1440, 960);w.show();w.page.setAudioMuted(True)
                until("document.querySelector('select[aria-label=选择算法]')")
                js("const b=document.createElement('div');b.textContent='秋叶的自动验证窗口 · 完成后自动关闭';b.style.cssText='position:fixed;top:4px;left:38%;z-index:99999;padding:6px 16px;background:#ffeab0;color:#342b22;pointer-events:none';document.body.append(b)")
                click('选择图片目录')
                until("document.querySelector('select[aria-label=语谱图图片]').options.length===2")
                file_id=js("document.querySelector('select[aria-label=语谱图图片]').options[1].value")
                select('语谱图图片',file_id)
                until("document.querySelector('.image-frame img')?.naturalWidth>0")
                select('选择算法','neural_pipeline')
                fill('时长（s）',.2);fill('频率终点（Hz）',8000)
                click('预测窗长与 dB 范围')
                until("document.body.innerText.includes('预测窗长')&&!document.body.innerText.includes('正在预测参数')")
                assert js("document.body.innerText.includes('参数预测已内置，可离线使用')")
                if not installed:
                    assert js("[...document.querySelectorAll('button')].find(e=>e.textContent.trim()==='开始重建').disabled")
                    assert not (case/'absent').exists()
                    for theme in ('light','dark'):
                        js('document.documentElement.dataset.theme='+json.dumps(theme));pause(300)
                        w.view.grab().save(str(out/('offline-'+theme+'.png')))
                    report['cases'].append(dict(kind='offline-prediction',ui=js("document.querySelector('.m09-settings').innerText")))
                    select('选择算法','griffin_lim')
                    until("document.body.innerText.includes('预测窗长')&&!document.body.innerText.includes('正在预测参数')")
                    click('开始重建')
                    until("document.querySelector('.comparison img')&&!document.querySelector('.m09-page[aria-busy=true]')")
                    click('保存重建结果');until("document.body.innerText.includes('已保存 4 个结果文件')")
                    meta=json.loads(next(saved.glob('*.json')).read_text('utf-8'))
                    assert meta['phase_method']=='griffin-lim' and meta['schema_version']=='m09-calibrated/1'
                    assert meta['config']['auto_parameters'] is False and 'vocoder' not in meta
                    report['cases'].append(dict(kind='offline-gl-prediction-and-render',metadata=meta))
                    w.view.grab().save(str(out/'offline-gl-reconstructed.png'))

                else:
                    until("document.body.innerText.includes('已下载，可离线使用')")
                    click('选择已有 Python 环境');click('检查兼容性')
                    until("document.body.innerText.includes('检查通过，可使用')",120)
                    for name,window,iterations in [('gpu-predicted',None,32),('gpu-manual-random-256',14,256)]:
                        if window is not None:
                            fill('窗长（ms）',window);fill('最小 dB（白）',-60);fill('最大 dB（黑）',0)
                            fill('迭代次数',iterations);select('Aoki 短时窗初始化','random')
                        print('TASK '+name,flush=True)
                        until("[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='开始重建'&&!e.disabled)",15)
                        click('开始重建')
                        until("document.querySelector('.comparison img')&&!document.querySelector('.m09-page[aria-busy=true]')",180)
                        click('保存重建结果')
                        until("document.body.innerText.includes('已保存 4 个结果文件')")
                        pause(100)
                        metas=[json.loads(p.read_text('utf-8')) for p in saved.glob('*.json')]
                        meta=next(m for m in metas if m['n_iter']==iterations)
                        assert meta['phase_method']=='griffin-lim+repair80+bigvgan'
                        assert meta['hop_length']*4==meta['win_length'] and meta['n_iter']==iterations
                        if window is None:
                            assert meta['config']['auto_parameters'] is False and meta['vocoder']['inference_device']=='cuda'
                        else:
                            assert meta['phase_initialization']=='random' and meta['vocoder']['inference_device']=='cuda'
                            assert meta['config']['auto_parameters'] is False
                            assert meta['window_ms']==window and meta['applied_config']['min_db']==-60
                        report['cases'].append(dict(kind=name,metadata=meta))
                    for p in saved.glob('*.wav'):
                        y,sr=sf.read(p);assert len(y)==round(.2*sr) and np.isfinite(y).all() and np.max(np.abs(y))<=.951
                    assert len(list(saved.glob('*.wav')))==2
                    imgs=[cv2.imread(str(p),0).shape for p in saved.glob('*.png')];assert all(shape==gray.shape for shape in imgs)
                    w.view.grab().save(str(out/'reconstructed.png'))
                    report['renderer']=js("(()=>{const g=document.createElement('canvas').getContext('webgl');const e=g?.getExtension('WEBGL_debug_renderer_info');return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):null})()")
                close_window()
            report['success']=True
        finally:
            if w is not None:w.view.grab().save(str(out/'last-state.png'))
            close_window();os.environ['PTB_VOCODER_ROOT']=existing_component_root
            write_json(out/'report.json',report);os.environ.pop('PTB_M09_PYTHON_CONFIG',None);app.quit()
        print('REPORT='+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
