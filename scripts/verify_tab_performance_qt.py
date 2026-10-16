"""Owned visible Qt WebEngine regression against the source component fixture."""
import argparse
import os
from pathlib import Path
import time

from verification_artifacts import verification_run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')
    from PyQt6.QtCore import QEventLoop, QTimer, QUrl
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtWebEngineCore import QWebEngineProfile, QWebEnginePage
    from PyQt6.QtWebEngineWidgets import QWebEngineView

    app = QApplication(['Tab-performance-owned-QA'])
    with verification_run('tab-performance', 'qt', 'Visible default-GPU Qt WebEngine; source six-tab 60s PCM/12000-point fixture and AppShell M01 preservation') as (out, scratch):
        view = QWebEngineView()
        profile = QWebEngineProfile(view)
        page = QWebEnginePage(profile, view)
        view.setPage(page)
        view.resize(1440, 1000)
        view.show()
        report = {'success': False, 'cases': []}

        def pause(ms=160):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, result = QEventLoop(), []
            page.runJavaScript(code, lambda value: (result.append(value), loop.quit()))
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            assert result, 'Qt JavaScript timeout'
            return result[0]

        def until(expression):
            deadline = time.monotonic() + 30
            while not js(expression):
                assert time.monotonic() < deadline, expression
                pause(50)

        try:
            view.load(QUrl(args.url))
            until('!!window.qa && document.querySelectorAll(".scientific-trace path").length===6')
            for index in range(6):
                js(f'qa.active.value={index}')
                pause()
            before = js('({requests:qa.counts.requests.length,paths:qa.counts.paths})')
            for theme in ['light', 'dark']:
                js(f'document.documentElement.dataset.theme="{theme}"')
                for index in range(6):
                    js(f'qa.active.value={index}')
                    pause()
                view.grab().save(str(out / f'{theme}.png'))
            after = js('({requests:qa.counts.requests.length,paths:qa.counts.paths})')
            assert after == before, (before, after)
            for _ in range(10):
                js('qa.cursor.value+=.1')
                pause(20)
            assert js('qa.counts.paths') == after['paths']
            report['cases'].append({'tabs': 6, 'switches': 12, 'before': before, 'after': after, 'cursor_updates': 10})
            view.load(QUrl(args.url + '?shell#M01'))
            until('!!window.qa && !!document.querySelector(".m01-file-list button")')
            js('document.querySelector(".m01-file-list .file-row").click()')
            until('!!document.querySelector(".wave-line")')
            js('qa.originalWave=document.querySelector(".wave-line")')
            for title in ['参数显示', 'EGG 信号分析', 'LPC 谱图']:
                js(f'[...document.querySelectorAll(".nav-item")].find(b=>b.textContent.trim()==="{title}").click()')
                pause()
                lists = js('qa.counts.lists')
                js('document.querySelector("#tab-M01").click()')
                pause()
                assert js('qa.counts.lists') == lists
                assert js('qa.originalWave===document.querySelector(".wave-line")')
            report['cases'].append({'shell_return_switches': 3, 'directory_rescans': 0, 'wave_dom_preserved': True})
            report['success'] = True
        finally:
            write_json(out / 'report.json', report)
            view.close()
            page.deleteLater()
            pause(100)
        print(str(out), report)


if __name__ == '__main__':
    main()
