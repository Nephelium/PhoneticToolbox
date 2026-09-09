"""P04 shared static frontend preview, using the P01 custom-scheme approach."""
import argparse
import json
import mimetypes
import sys
from pathlib import Path

from PyQt6.QtCore import QBuffer, QIODevice, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QIcon, QRawFont
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineCore import (QWebEnginePage, QWebEngineProfile, QWebEngineSettings,
    QWebEngineUrlRequestJob, QWebEngineUrlScheme, QWebEngineUrlSchemeHandler, QWebEngineUrlRequestInterceptor)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from probe_paths import resolve_asset

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / 'frontend/dist'
scheme = QWebEngineUrlScheme(b'ptbapp')
scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
scheme.setFlags(QWebEngineUrlScheme.Flag.SecureScheme | QWebEngineUrlScheme.Flag.LocalScheme |
                QWebEngineUrlScheme.Flag.LocalAccessAllowed | QWebEngineUrlScheme.Flag.CorsEnabled |
                QWebEngineUrlScheme.Flag.FetchApiAllowed)
QWebEngineUrlScheme.registerScheme(scheme)


class Assets(QWebEngineUrlSchemeHandler):
    def requestStarted(self, job):
        if job.requestUrl().host() != 'app' or bytes(job.requestMethod()) != b'GET':
            job.fail(QWebEngineUrlRequestJob.Error.RequestDenied)
            return
        try:
            path = resolve_asset(DIST, job.requestUrl().path())
            content = QBuffer(job)
            content.setData(path.read_bytes())
            content.open(QIODevice.OpenModeFlag.ReadOnly)
            mime = {'.js': 'text/javascript', '.css': 'text/css', '.wav': 'audio/wav'}.get(
                path.suffix, mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            job.reply(mime.encode('ascii'), content)
        except (ValueError, OSError):
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)


class LocalOnly(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, info):
        url = info.requestUrl()
        if not (url.scheme() == 'ptbapp' and url.host() == 'app') and url.scheme() not in {'data', 'blob'}:
            info.block(True)


class Page(QWebEnginePage):
    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        if url.scheme() == 'https':
            QDesktopServices.openUrl(url)
            return False
        return url.scheme() == 'ptbapp' and url.host() == 'app'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', type=Path)
    args = parser.parse_args()
    if not (DIST / 'index.html').exists():
        raise SystemExit('Build the shared frontend first: npm --prefix frontend run build')
    app = QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle('PhoneticToolbox 3.0 · 工作台试用')
    window.resize(1440, 900)
    window.setWindowIcon(QIcon(str(ROOT / 'frontend/src/assets/k2.png')))
    profile = QWebEngineProfile(window) if args.self_test else QWebEngineProfile('ptb-p04-preview', window)
    if not args.self_test:
        profile.setPersistentStoragePath(str(ROOT / 'output/p04-profile'))
        profile.setCachePath(str(ROOT / 'output/p04-profile/cache'))
    assets, interceptor = Assets(profile), LocalOnly(profile)
    profile.installUrlSchemeHandler(b'ptbapp', assets)
    profile.setUrlRequestInterceptor(interceptor)
    view = QWebEngineView(window)
    page = Page(profile, view)
    page.permissionRequested.connect(lambda request: request.deny())
    page.newWindowRequested.connect(lambda request: QDesktopServices.openUrl(request.requestedUrl())
                                    if request.requestedUrl().scheme() == 'https' else None)
    page.settings().setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, True)
    view.setPage(page)
    window.setCentralWidget(view)
    if args.self_test:
        output = args.self_test.resolve()
        assert output.is_relative_to(ROOT / 'output/validation/p04')
        output.mkdir(parents=True, exist_ok=True)
        layouts = []
        scales = [1.0, 1.25, 1.5, 2.0]
        font = QRawFont(str(ROOT / 'frontend/src/assets/DoulosSIL-Regular.ttf'), 24)
        missing = [hex(ord(c)) for c in 'aɑəɚɤɿʅŋɲʂʐtʰʈʂʰ˥˩' if not font.supportsCharacter(ord(c))]
        def finish(snapshot):
            scale = scales[len(layouts)]
            layouts.append(dict(snapshot or {}, zoom=scale))
            view.grab().save(str(output / f'qt-home-{scale}.png'))
            if len(layouts) < len(scales):
                view.setZoomFactor(scales[len(layouts)])
                QTimer.singleShot(350, inspect)
                return
            good = font.isValid() and not missing and all(s.get('modules') == 15 and s.get('home')
                and s.get('desktop') and not s.get('overflow') for s in layouts)
            (output / 'qt-summary.json').write_text(json.dumps({'success': good, 'layouts': layouts,
                'font_missing_codepoints': missing,
                'scope': 'Qt custom-scheme static build and WebEngine zoom 100/125/150/200%; not OS display-setting changes'},
                ensure_ascii=False, indent=2), encoding='utf-8')
            app.exit(0 if good else 1)
        def inspect():
            page.runJavaScript(
                "({modules:document.querySelectorAll('.nav-group .nav-item').length,"
                "home:!!document.querySelector('.welcome'),desktop:document.querySelector('.host-badge')?.textContent==='本地桌面',"
                "width:innerWidth,height:innerHeight,dpr:devicePixelRatio,"
                "overflow:document.querySelector('main').scrollWidth>document.querySelector('main').clientWidth})", finish)
        def loaded(ok):
            if not ok:
                app.exit(1)
                return
            QTimer.singleShot(500, inspect)
        view.loadFinished.connect(loaded)
        QTimer.singleShot(15000, lambda: app.exit(2))
    window.show()
    view.load(QUrl('ptbapp://app/index.html'))
    code = app.exec()
    page.deleteLater()
    app.processEvents()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
