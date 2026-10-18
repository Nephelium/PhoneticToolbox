"""Owned LPC numeric/Qt regression; an optional named private WAV is read only."""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from verification_artifacts import verification_run, write_json
from workbench_source import configure, bind_sources

ROOT = Path(__file__).resolve().parents[1]


def numeric(out, source):
    bind_sources(ROOT)
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import find_peaks
    from phonetic_core.lpc import compute_spectrum, LPCConfig, mono_samples
    original = ROOT / 'phonetic_toolbox/core/acoustic/lpc.py'
    frozen = json.loads((ROOT / 'tests/fixtures/m04/manifest.json').read_text('utf-8'))
    assert hashlib.sha256(original.read_bytes()).hexdigest() == frozen['source_hashes']['phonetic_toolbox/core/acoustic/lpc.py']
    spec = importlib.util.spec_from_file_location('frozen_v2_lpc', original)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    fs, data = wavfile.read(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    selected = data[int(1.552971 * fs):int(1.65 * fs)]
    report = dict(source_sha256=before, rate=int(fs), frames=len(data), channels=data.shape[1], roi_samples=len(selected), cases=[])
    for name, samples in [('mean', selected)] + [(str(i+1), selected[:, i]) for i in range(data.shape[1])]:
        audio = mono_samples(samples)
        result = compute_spectrum(audio, int(fs), LPCConfig(freq_max_hz=5000, dynamic_y=True))
        frequency, magnitude = old.compute_lpc_spectrum(audio, int(fs), 50)
        assert frequency.tobytes() == result.frequencies_hz.tobytes()
        assert magnitude.tobytes() == result.magnitude_db.tobytes()
        peaks = find_peaks(magnitude)[0]; peaks = peaks[frequency[peaks] <= 5000]
        report['cases'].append(dict(channel=name, v2_byte_exact=True, mean=float(audio.mean()), rms=float(np.sqrt(np.mean(audio**2))),
            dc_db=float(magnitude[0]), y=[result.amp_min_db, result.amp_max_db], peaks_hz=frequency[peaks].tolist()))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    write_json(out / 'natural-audio.json', report)


def compare_browser(source):
    """The same generated WAV bytes, with the existing cross-runtime tolerance."""
    bind_sources(ROOT)
    import io
    import numpy as np
    from scipy.io import wavfile
    from phonetic_core.lpc import LPCConfig, compute_spectrum, mono_samples
    from phonetic_core.lpc.spectrum import fft_reference_spectrum
    data=json.loads(source.read_text('utf-8')); raw=base64.b64decode(data['raw']); fs,samples=wavfile.read(io.BytesIO(raw))
    report=[]; passed=True
    for key,channel in [('mean',None),('selected',0)]:
        metadata=data[key]; selection=metadata['selection']; selected=samples[selection['start_sample']:selection['end_sample']]
        audio=mono_samples(selected if channel is None else selected[:,channel]); actual=compute_spectrum(audio,int(fs),LPCConfig())
        assert hashlib.sha256(raw).hexdigest()==metadata['input_sha256']
        np.testing.assert_array_equal(actual.frequencies_hz,metadata['spectrum']['frequencies_hz'])
        frequency,magnitude=fft_reference_spectrum(audio,int(fs))
        np.testing.assert_array_equal(frequency,metadata['reference_spectrum']['frequencies_hz'])
        row=dict(channel=key)
        for name,values,expected in [('lpc',actual.magnitude_db,metadata['spectrum']['magnitude_db']),('fft',magnitude,metadata['reference_spectrum']['magnitude_db'])]:
            expected=np.array(expected); delta=abs(values-expected); violations=delta>(1e-10+1e-7*abs(expected))
            row[name]=dict(max_abs=float(delta.max()),violations=int(violations.sum()),failed_points=[dict(index=int(i),actual=float(values[i]),expected=float(expected[i]),difference=float(delta[i])) for i in np.flatnonzero(violations)])
            try: np.testing.assert_allclose(values,expected,rtol=1e-7,atol=1e-10)
            except AssertionError: passed=False
        report.append(row)
    print(json.dumps(dict(success=passed,cases=report,rtol=1e-7,atol=1e-10)),flush=True)
    assert passed, 'Strict cross-runtime comparison failed; see per-array differences above. LPC implementation remains unchanged.'


def qt(out):
    configure(ROOT)
    import numpy as np
    from scipy.io import wavfile
    from PyQt6.QtCore import QEvent, QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from PyQt6.QtGui import QImage
    from PyQt6 import sip
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    scratch = out / 'scratch'; inputs = scratch / 'inputs'; inputs.mkdir()
    saved = scratch / 'saved'; saved.mkdir(); cache = scratch / 'cache'; cache.mkdir(); initialize_local_files(cache)
    db = scratch / 'jobs.sqlite3'
    with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro', uri=True) as source:
        assert source.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()[0] == 0
        with sqlite3.connect(db) as target: source.backup(target)
    rate = 16000; t = np.arange(3200) / rate
    samples = np.column_stack([.2*np.sin(2*np.pi*440*t)+.1*np.sin(2*np.pi*1200*t), .4+.2*np.sin(2*np.pi*3200*t)])
    wavfile.write(inputs / 'stereo.wav', rate, samples)
    register_scheme(); app = QApplication(['LPC-reference-owned-QA']); app.setQuitOnLastWindowClosed(False)
    report = dict(success=False, checks=[], layouts=[], scope='Windows source Qt; synthetic stereo; real native task and export')
    window = None

    def pause(ms=80):
        loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()

    def js(code):
        loop = QEventLoop(); values = []
        window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
        QTimer.singleShot(8000, loop.quit); loop.exec(); assert values, 'JavaScript timeout'
        return values[0]

    def until(code, seconds=60):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js('Boolean('+code+')'): return
            pause()
        raise AssertionError(code+' / '+str(js('document.body.innerText.slice(-2200)')))

    def click(label):
        assert js("(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&e.textContent.trim()==="+json.dumps(label)+");if(!e||e.disabled)return false;e.click();return true})()"), label
        pause()

    def select(label, value):
        js("(()=>{const e=document.querySelector('select[aria-label=\""+label+"\"]');e.value="+json.dumps(value)+";e.dispatchEvent(new Event('change',{bubbles:true}));})()")
        pause()

    try:
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache,
                           vocal_profile=scratch / 'vocal', start_module='M04')
        window.resize(1440, 1000); window.show(); window.page.setAudioMuted(True)
        QFileDialog.getExistingDirectory = lambda *args, **kwargs: str(saved if '结果' in str(args) else inputs)
        QFileDialog.getSaveFileName = lambda *args, **kwargs: (str(saved / 'overlay.png'), '')
        until("!!document.querySelector('.lpc-page')"); click('打开音频目录')
        until("document.querySelector('select[aria-label=\"LPC 音频文件\"]').options.length>1")
        audio_id = js("document.querySelector('select[aria-label=\"LPC 音频文件\"]').options[1].value")
        select('LPC 音频文件', audio_id)
        until("!!document.querySelector('.wave-track svg')&&!document.querySelector('.lpc-files').innerText.includes('正在读取')")
        assert js("document.querySelector('select[aria-label=\"LPC 分析声道\"]').value") == 'mean'
        click('开始分析'); until("!!document.querySelector('.lpc-spectrum svg')")
        mean_path = js("document.querySelector('.lpc-spectrum .scientific-trace path').getAttribute('d')")
        click('显示频谱图'); until("document.querySelectorAll('.lpc-spectrum .scientific-trace').length===2")
        assert js("document.querySelectorAll('.lpc-spectrum .scientific-trace path')[1].getAttribute('d')") == mean_path
        click('显示频谱图'); assert js("document.querySelectorAll('.lpc-spectrum .scientific-trace').length") == 1
        report['checks'].append('FFT toggle preserves LPC path and creates no new task')
        select('LPC 分析声道', '0'); until("document.querySelector('.lpc-page .notice')?.textContent.includes('已改变')")
        click('开始分析'); until("document.querySelector('.lpc-spectrum+ p')?.textContent.includes('分析声道 1')")
        assert js("document.querySelector('.lpc-spectrum .scientific-trace path').getAttribute('d')") != mean_path
        click('显示频谱图'); click('动态纵轴')
        click('保存 PNG 图片'); until("document.querySelector('.lpc-results').textContent.includes('保存')")
        deadline = time.monotonic()+20
        while not (saved/'overlay.png').exists() and time.monotonic()<deadline: pause()
        image = QImage(str(saved/'overlay.png')); assert (image.width(), image.height()) == (2400, 1350)
        assert abs(image.dotsPerMeterX()*.0254-300) < .01
        image.save(str(out/'overlay.png'))
        click('选择目录保存完整结果')
        until("document.querySelector('.lpc-page').innerText.includes('已保存 3 个结果')")
        metadata = json.loads(next(saved.glob('LPC*.json')).read_text('utf-8'))
        assert metadata['config']['analysis_channel'] == 0
        _, clip = wavfile.read(next(saved.glob('LPC*.wav')))
        np.testing.assert_array_equal(clip, samples[:, 0])
        assert len(metadata['reference_spectrum']['frequencies_hz']) == 2049
        report['checks'].append('Selected channel snapshot, exact saved WAV/FFT, native 2400x1350 300dpi overlay PNG')
        for width, height in ((1440, 1000), (1100, 800)):
            window.resize(width, height); pause(250)
            for theme in ('light', 'dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme)); pause(150)
                name=f'qt-{width}-{theme}'; shot=window.view.grab().toImage()
                colors={shot.pixelColor(x,y).rgba() for x in range(0,shot.width(),32) for y in range(0,shot.height(),32)}
                assert len(colors)>10, 'Qt view has not rendered'
                shot.save(str(out/(name+'.png')))
                report['layouts'].append(dict(name=name, traces=js("document.querySelectorAll('.lpc-spectrum .scientific-trace').length")))
        assert all(item['traces'] == 2 for item in report['layouts'])
        report['success'] = True
    finally:
        write_json(out/'qt-report.json', report)
        if window:
            js('window.onbeforeunload=null'); window.closing=True; window.close(); window.service.close(); pause(400); window.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete); pause(200)
        app.quit(); sip.delete(app)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--audio', type=Path); parser.add_argument('--child', type=Path); parser.add_argument('--numeric', type=Path); parser.add_argument('--compare-browser',type=Path)
    args=parser.parse_args()
    if args.child: qt(args.child)
    elif args.numeric: numeric(args.numeric, args.audio)
    elif args.compare_browser: compare_browser(args.compare_browser)
    else:
        with verification_run('m04-reference', 'qt', '16000 Hz, 3200-frame two-channel FLOAT64; real Qt/worker/native save; optional read-only named natural WAV') as (out, scratch):
            if args.audio:
                prefix=ROOT/'.venv/m03-compatible'; env=os.environ.copy()
                env['PATH']=os.pathsep.join([str(prefix),str(prefix/'Library/bin'),env['PATH']])
                result=subprocess.run([str(prefix/'python.exe'),'-B','-X','utf8',__file__,'--numeric',str(out),'--audio',str(args.audio)],env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
                (out/'numeric.log').write_text(result.stdout+'\n'+result.stderr,'utf-8'); assert result.returncode==0, 'See numeric.log'
            result=subprocess.run([sys.executable,'-B','-X','utf8',__file__,'--child',str(out)],capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
            (out/'qt.log').write_text(result.stdout+'\n'+result.stderr,'utf-8'); print(out,flush=True)
            assert result.returncode==0, 'See qt.log'
