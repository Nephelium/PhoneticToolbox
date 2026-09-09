"""P01 isolated Qt WebEngine host. No scientific backend or network server."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import mimetypes
import os
import platform
import sys
import time
from pathlib import Path

if os.environ.get('PTB_QT_BINDING') == 'PySide6':
    from PySide6.QtCore import QBuffer, QIODevice, QTimer, QUrl, qVersion
    from PySide6.QtGui import QIcon, QRawFont
    from PySide6.QtMultimedia import QMediaDevices
    from PySide6.QtWidgets import QApplication, QMainWindow
    from PySide6.QtWebEngineCore import (QWebEnginePage, QWebEngineProfile, QWebEngineSettings,
        QWebEngineUrlRequestInterceptor, QWebEngineUrlRequestJob, QWebEngineUrlScheme, QWebEngineUrlSchemeHandler)
    from PySide6.QtWebEngineWidgets import QWebEngineView
    BINDING = 'PySide6'
else:
    from PyQt6.QtCore import QBuffer, QIODevice, QTimer, QUrl, qVersion
    from PyQt6.QtGui import QIcon, QRawFont
    from PyQt6.QtMultimedia import QMediaDevices
    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtWebEngineCore import (QWebEnginePage, QWebEngineProfile, QWebEngineSettings,
        QWebEngineUrlRequestInterceptor, QWebEngineUrlRequestJob, QWebEngineUrlScheme, QWebEngineUrlSchemeHandler)
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    BINDING = 'PyQt6'

from probe_paths import resolve_asset

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2] if not getattr(sys, 'frozen', False) else Path(sys._MEIPASS)
ASSETS = ROOT / ('web' if getattr(sys, 'frozen', False) else 'frontend/experiments/audio-viewport/dist')

scheme = QWebEngineUrlScheme(b'ptbprobe')
scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
scheme.setFlags(QWebEngineUrlScheme.Flag.SecureScheme | QWebEngineUrlScheme.Flag.CorsEnabled
                | QWebEngineUrlScheme.Flag.FetchApiAllowed)
QWebEngineUrlScheme.registerScheme(scheme)


class StaticAssets(QWebEngineUrlSchemeHandler):
    def requestStarted(self, job):
        if job.requestUrl().host() != 'app' or bytes(job.requestMethod()) != b'GET':
            job.fail(QWebEngineUrlRequestJob.Error.RequestDenied)
            return
        try:
            path = resolve_asset(ASSETS, job.requestUrl().path())
            content = QBuffer(job)
            content.setData(path.read_bytes())
            content.open(QIODevice.OpenModeFlag.ReadOnly)
            mime = {'.js': 'text/javascript', '.css': 'text/css', '.ttf': 'font/ttf', '.wav': 'audio/wav'}.get(
                path.suffix, mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            job.reply(mime.encode('ascii'), content)
        except (ValueError, OSError):
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)


class LocalOnly(QWebEngineUrlRequestInterceptor):
    def __init__(self, parent):
        super().__init__(parent)
        self.blocked = []

    def interceptRequest(self, info):
        url = info.requestUrl()
        if not (url.scheme() == 'ptbprobe' and url.host() == 'app') and url.scheme() != 'data':
            self.blocked.append(url.scheme())
            info.block(True)


class Page(QWebEnginePage):
    def __init__(self, profile, parent):
        super().__init__(profile, parent)
        self.result_callback = None
        self.messages = []

    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        return url.scheme() == 'ptbprobe' and url.host() == 'app' and url.path() == '/index.html'

    def javaScriptConsoleMessage(self, level, message, line, source):
        if message.startswith('PTB_PROBE_JSON:') and self.result_callback:
            self.result_callback(json.loads(message[len('PTB_PROBE_JSON:'):]))
        else:
            self.messages.append({'level': str(level), 'message': message, 'line': line})


class Host(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle(f'PhoneticToolbox · P01 音频原型 ({BINDING})')
        self.resize(1280, 860)
        self.setMinimumSize(560, 520)
        self.setWindowIcon(QIcon(str(ASSETS / 'icon.png')))
        self.profile = QWebEngineProfile(self)
        self.profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
        self.assets = StaticAssets(self.profile)
        self.profile.installUrlSchemeHandler(b'ptbprobe', self.assets)
        self.interceptor = LocalOnly(self.profile)
        self.profile.setUrlRequestInterceptor(self.interceptor)
        self.view = QWebEngineView(self)
        self.page = Page(self.profile, self.view)
        self.page.permissionRequested.connect(lambda permission: permission.deny())
        self.page.settings().setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, True)
        self.view.setPage(self.page)
        self.view.setAcceptDrops(False)
        self.setCentralWidget(self.view)
        self.page.renderProcessTerminated.connect(lambda status, code: self.fail(f'Renderer terminated: {status}, {code}'))
        self.view.loadFinished.connect(self.loaded)
        self.report = {'binding': BINDING, 'qt': qVersion(), 'python': platform.python_version(),
                       'platform': platform.platform(), 'frozen': bool(getattr(sys, 'frozen', False)),
                       'pid': os.getpid(), 'runtime_root': str(ROOT), 'cwd': str(Path.cwd()),
                       'network': 'blocked by host; custom local scheme', 'checks': {}}
        self.output = Path(args.self_test).resolve() if args.self_test else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
        self.timeout = QTimer(self)
        self.timeout.setSingleShot(True)
        self.timeout.timeout.connect(lambda: self.fail('Self-test timed out'))
        if self.output:
            self.timeout.start(45000)
        self.show()
        self.view.load(QUrl('ptbprobe://app/index.html' + ('?probe=1' if self.output else '')))

    def write_report(self):
        if self.output:
            self.report['console'] = self.page.messages
            self.report['blocked_schemes'] = self.interceptor.blocked
            (self.output / 'host-report.json').write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def fail(self, message):
        self.report['success'] = False
        self.report['error'] = message
        self.write_report()
        QApplication.exit(1)

    def loaded(self, ok):
        if not ok:
            self.fail('Built page did not load')
            return
        self.report['loaded_seconds'] = round(time.monotonic() - STARTED, 3)
        if self.output:
            QTimer.singleShot(350, self.begin_test)

    def begin_test(self):
        self.activateWindow()
        self.view.setFocus()
        self.view.grab().save(str(self.output / 'empty-light.png'))
        font = QRawFont(str(ASSETS / 'DoulosSIL-Regular.ttf'), 24)
        self.report['font'] = {'valid': font.isValid(), 'family': font.familyName(),
            'missing_codepoints': [f'U+{ord(c):04X}' for c in 'pʰaːtɕʰiŋ˨˩˦ɹ̩ɚn̩ãɡ͡b' if not font.supportsCharacter(ord(c))]}
        self.report['audio_outputs'] = [{'name': device.description(), 'default': device.isDefault(),
            'sample_rate': device.preferredFormat().sampleRate(), 'channels': device.preferredFormat().channelCount()}
            for device in QMediaDevices.audioOutputs()]
        self.page.result_callback = self.audio_test_done
        # QTest mouse press exercises the actual button and gives Chromium a user gesture.
        if BINDING == 'PySide6':
            from PySide6.QtCore import QPoint, Qt
            from PySide6.QtTest import QTest
        else:
            from PyQt6.QtCore import QPoint, Qt
            from PyQt6.QtTest import QTest
        self.page.runJavaScript("(()=>{const r=document.querySelector('#fixture').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()",
            lambda point: QTest.mouseClick(self.view.focusProxy(), Qt.MouseButton.LeftButton, pos=QPoint(round(point[0]), round(point[1])), delay=50))
        QTimer.singleShot(600, lambda: self.page.runJavaScript(TEST_SCRIPT))

    def audio_test_done(self, result):
        self.report['audio_test'] = result
        if not result.get('success'):
            self.fail(result.get('error', 'Audio test failed'))
            return
        QTimer.singleShot(250, self.capture_light)

    def capture_light(self):
        self.view.grab().save(str(self.output / 'waveform-light.png'))
        self.page.result_callback = self.layout_done
        self.page.runJavaScript("window.__probe.setTheme(true)")
        QTimer.singleShot(200, lambda: self.capture_layout(1280, 800, 'dark-1280'))

    def capture_layout(self, width, height, name):
        self.resize(width, height)
        QTimer.singleShot(250, lambda: self.layout_snapshot(name))

    def layout_snapshot(self, name):
        self.view.grab().save(str(self.output / f'{name}.png'))
        self.page.runJavaScript("console.info('PTB_PROBE_JSON:'+JSON.stringify({name:" + json.dumps(name) + ",snapshot:window.__probe.snapshot()}))")

    def layout_done(self, result):
        self.report.setdefault('layouts', []).append(result)
        if result['name'] == 'dark-1280':
            self.capture_layout(960, 720, 'dark-960')
        elif result['name'] == 'dark-960':
            self.page.runJavaScript('window.__probe.setTheme(false)')
            self.capture_layout(640, 700, 'light-640')
        else:
            self.finish()

    def finish(self):
        self.report['checks'] = {
            'audio_contracts': self.report['audio_test']['success'],
            'font_glyphs': self.report['font']['valid'] and not self.report['font']['missing_codepoints'],
            'audio_output_enumerated': bool(self.report['audio_outputs']),
            'layouts_no_horizontal_overflow': all(not row['snapshot']['overflow'] for row in self.report['layouts']),
            'selection_survives_layout': all(row['snapshot']['selection'] == [123, 22173] for row in self.report['layouts']),
        }
        self.report['success'] = all(self.report['checks'].values())
        self.timeout.stop()
        self.write_report()
        QTimer.singleShot(200, lambda: QApplication.exit(0 if self.report['success'] else 1))


TEST_SCRIPT = r"""
(async () => {
 const p=window.__probe, checks={}, snapshots=[];
 const wait=ms=>new Promise(r=>setTimeout(r,ms));
 const check=(key,condition)=>{checks[key]=!!condition;if(!condition)throw Error(key)};
 try {
   for(let i=0;i<30&&!p.snapshot().sampleCount;i++)await wait(100);
   await document.fonts.load('24px "Doulos SIL"');
   const initial=p.snapshot(); snapshots.push(initial);
   check('fixture_44100_stereo_88200',initial.sampleRate===44100&&initial.sampleCount===88200&&initial.channels===2);
   check('no_autoplay',!initial.playing);
   p.select(123,22173);
   const offline=await p.offlineCheck();
   check('offline_exact_22050_frames',offline.frames===22050&&offline.sampleRate===44100&&offline.channels===2&&offline.maxError===0);
   p.zoomSelection();check('zoom_samples',JSON.stringify(p.snapshot().viewport)==='[123,22173]');p.fit();
   p.select(88200,88200);await p.play();check('empty_selection_silent',!p.snapshot().playing);
   p.select(0,88200);
   // Actual click is dispatched by Qt below; script supplies a handler after user activation exists.
   await p.play();
   for(let i=0;i<40&&p.snapshot().cursor===0;i++)await wait(50);
   await wait(700);snapshots.push(p.snapshot());
   check('web_audio_running',p.snapshot().playing&&p.snapshot().contextState==='running'&&p.snapshot().cursor>0);
   p.pause();const pause=p.snapshot();await wait(100);
   check('pause_cursor_stable',pause.paused&&!pause.playing&&p.snapshot().cursor===pause.cursor);
   await p.play();await wait(600);check('resume_advances',p.snapshot().cursor>pause.cursor);
   p.stop();check('stop_resets',!p.snapshot().playing&&p.snapshot().cursor===0);
   p.select(88199,88200);const last=await p.offlineCheck();check('last_frame',last.frames===1&&last.maxError===0);
   p.select(123,22173);snapshots.push(p.snapshot());
   console.info('PTB_PROBE_JSON:'+JSON.stringify({success:true,checks,offline,last,snapshots}));
 } catch(e) {p?.stop();console.info('PTB_PROBE_JSON:'+JSON.stringify({success:false,error:String(e),checks,snapshots}));}
})()
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', metavar='OUTPUT_DIRECTORY')
    args = parser.parse_args()
    if not (ASSETS / 'index.html').is_file():
        raise SystemExit('Build frontend/experiments/audio-viewport before starting P01.')
    app = QApplication(sys.argv[:1])
    app.setApplicationName('PhoneticToolbox-P01')
    window = Host(args)
    code = app.exec()
    # Explicit Qt ownership order: destroy page before its off-the-record profile.
    window.view.setPage(QWebEnginePage(window.view))
    window.page.deleteLater()
    app.processEvents()
    return code


if __name__ == '__main__':
    sys.exit(main())
