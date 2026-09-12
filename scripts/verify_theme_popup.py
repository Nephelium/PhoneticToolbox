"""P04 theme popup regression: real Qt page, no API/worker or user profile."""
import json
from pathlib import Path
from uuid import uuid4
from PyQt6.QtCore import QUrl, Qt, QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from ptb_desktop.host import register_scheme, Assets, LocalOnly

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'output/validation/theme-popup' / uuid4().hex
    out.mkdir(parents=True)
    register_scheme()
    app = QApplication(['P04 theme popup verification'])
    profile = QWebEngineProfile()
    assets = Assets(ROOT / 'frontend/dist', profile)
    profile.installUrlSchemeHandler(b'ptbapp', assets)
    guard = LocalOnly('http://127.0.0.1:0', profile)
    profile.setUrlRequestInterceptor(guard)
    page = QWebEnginePage(profile)
    view = QWebEngineView()
    view.setPage(page)
    view.setWindowTitle('P04 theme popup verification — no service')
    view.resize(1050, 760)
    view.show()
    page.setUrl(QUrl('ptbapp://app/'))

    def js(code):
        values = []
        page.runJavaScript(code, values.append)
        for _ in range(200):
            if values:
                return values[0]
            QTest.qWait(25)
        raise AssertionError('JavaScript timeout')

    report = []
    try:
        for _ in range(200):
            if js("!!document.querySelector('[aria-label=\"配色主题\"]')"):
                break
            QTest.qWait(25)
        else:
            raise AssertionError('Theme selector unavailable')
        for choice in ['dark', 'light', 'system']:
            expected = choice if choice != 'system' else js("matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'")
            js("(()=>{const s=document.querySelector('[aria-label=\"配色主题\"]');s.value=" + json.dumps(choice) + ";s.dispatchEvent(new Event('change',{bubbles:true}));})()")
            QTest.qWait(400)
            actual = js("(()=>{const s=document.querySelector('[aria-label=\"配色主题\"]'),r=s.getBoundingClientRect(),o=getComputedStyle([...s.options].find(o=>!o.selected));return {theme:document.documentElement.dataset.theme,bg:o.backgroundColor,fg:o.color,x:r.x+r.width/2,y:r.y+r.height/2};})()")
            assert actual['theme'] == expected, actual
            assert actual['bg'] == ('rgb(24, 36, 50)' if expected == 'dark' else 'rgb(255, 255, 255)'), actual
            tag = choice + '-' + expected
            view.setFocus()
            QTest.mouseClick(view.focusProxy(), Qt.MouseButton.LeftButton, pos=QPoint(round(actual['x']), round(actual['y'])))
            QTest.qWait(300)
            popups = [w for w in app.topLevelWindows() if w.isVisible() and w != view.windowHandle()]
            assert popups, 'Native popup did not open'
            popup = popups[-1]
            assert popup.screen().grabWindow(int(popup.winId())).save(str(out / (tag + '-popup.png')))
            # Dismiss without changing the chosen value, preserving ordinary keyboard behavior.
            QTest.keyClick(view.focusProxy(), Qt.Key.Key_Escape)
            QTest.qWait(100)
            report.append({'selection': choice, **actual})
        (out / 'report.json').write_text(json.dumps({'success': True, 'checks': report, 'services_started': False}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(out / 'report.json')
    finally:
        view.close()
        page.deleteLater()
        QTest.qWait(100)
        profile.deleteLater()
        QTest.qWait(100)


if __name__ == '__main__':
    main()
