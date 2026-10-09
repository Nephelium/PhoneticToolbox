"""M13 verification in the actual Windows Qt desktop host.

This opens the production frontend bundle through the local-only ptbapp scheme.
It does not submit a backend job or mutate the shared task database.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from uuid import uuid4

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication

from ptb_desktop.host import Workbench, register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out = (ROOT / "output/validation/m13-qt" / uuid4().hex).absolute()
    out.mkdir(parents=True)
    cache = out / "cache"
    cache.mkdir()
    initialize_local_files(cache)

    register_scheme()
    app = QApplication(["M13 Qt verification"])
    window = Workbench(
        ROOT / "frontend/dist",
        test=True,
        jobs_path=ROOT / "output/validation/p06/local-state.sqlite3",
        local_files_root=cache,
        vocal_resources=ROOT / "resources/vocal_tract/native",
    )
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(1440, 900)
    window.show()
    assert not app.windowIcon().isNull() and not window.windowIcon().isNull()
    # Inspect the actual HWND icon handles used by the caption and taskbar.
    import ctypes
    from ctypes import wintypes
    send = ctypes.windll.user32.SendMessageW
    send.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    send.restype = ctypes.c_void_p
    native_icons = {name: bool(send(int(window.winId()), 0x007F, kind, 0))
                    for name, kind in [('small', 0), ('large', 1)]}
    assert all(native_icons.values()), native_icons
    window.windowIcon().pixmap(64, 64).save(str(out / 'window-icon.png'))

    click_nav = "[...document.querySelectorAll('.nav-item')].find(b=>b.textContent.includes('汉字转国际音标'))?.click()"
    fill = "(()=>{const e=document.querySelector('[aria-label=\"待转换汉字文本\"]');e.value='银行花';e.dispatchEvent(new Event('input',{bubbles:true}));})()"
    select_pinyin = "(()=>{const e=document.querySelector('[aria-label=\"转换标准\"]');e.value='汉语拼音';e.dispatchEvent(new Event('change',{bubbles:true}));})()"
    stages = [
        ("document.querySelector('.host-badge')?.textContent==='本地桌面'", click_nav),
        ("!!document.querySelector('.mandarin-ipa-page')&&[...document.fonts].some(f=>f.family==='PTB-Doulos'&&f.status==='loaded')", fill),
        ("document.querySelector('[aria-label=\"待转换汉字文本\"]')?.value==='银行花'&&document.querySelectorAll('[aria-label=\"转换标准\"] option').length===11", select_pinyin),
        ("(()=>{const t=[...document.querySelectorAll('.m13-mapped')],tops=s=>t.map(e=>e.querySelector(s).getBoundingClientRect().top),i=tops('.m13-ipa'),h=tops('.m13-hanzi');return t[0]?.dataset.value==='yín'&&Math.max(...i)-Math.min(...i)<.5&&Math.max(...h)-Math.min(...h)<.5&&document.querySelectorAll('.global-transport').length===0&&document.querySelectorAll('.mandarin-ipa-page h1').length===0;})()", "document.documentElement.dataset.theme='light'"),
        ("document.documentElement.dataset.theme==='light'&&document.querySelector('.mandarin-ipa-page')?.textContent.includes('保存本机草稿 *')", "__light__"),
        ("document.querySelector('[aria-label=\"待转换汉字文本\"]')?.value==='银行花'", "document.documentElement.dataset.theme='dark'"),
        ("document.documentElement.dataset.theme==='dark'&&getComputedStyle(document.querySelector('.m13-ipa')).fontFamily.includes('PTB-Doulos')", "__dark__"),
        ("true", "document.querySelector('input[value=stacked]').click()"),
        ("(()=>{const r=s=>document.querySelector(s).getBoundingClientRect(),i=r('.m13-input-section'),o=r('.m13-result-section'),c=r('.m13-settings-section');return o.top>=i.bottom&&c.left>=o.right-1;})()", "document.querySelector('.m13-ambiguous').click()"),
        ("(()=>{const p=document.querySelector('.m13-variants')?.getBoundingClientRect(),a=document.querySelector('.m13-ambiguous[aria-expanded=true]')?.getBoundingClientRect();return p&&a&&Math.abs(p.width-p.height)<2&&p.right<=innerWidth&&p.bottom<=innerHeight&&p.top>=0;})()", "__dark__"),
    ]
    report = {
        "success": False,
        "platform": sys.platform,
        "host": "Qt WebEngine / ptbapp local-only scheme",
        "stages": [],
        "database_operations": "none",
        "native_icons": native_icons,
    }
    index = 0
    pending = False
    started = time.monotonic()

    def close_window() -> None:
        window.closing = True
        window.close()

    def finish(success: bool, error: str | None = None) -> None:
        timer.stop()
        report.update(success=success, error=error)

        def diagnostics(value: object) -> None:
            report["diagnostics"] = value
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            QTimer.singleShot(100, close_window)

        window.page.runJavaScript(
            """(()=>({
              input:document.querySelector('[aria-label="待转换汉字文本"]')?.value,
              standard:document.querySelector('[aria-label="转换标准"]')?.value,
              standards:document.querySelectorAll('[aria-label="转换标准"] option').length,
              mapped:[...document.querySelectorAll('.m13-mapped')].map(e=>e.dataset.value),
              tokenTops:[...document.querySelectorAll('.m13-mapped')].map(e=>({ipa:e.querySelector('.m13-ipa').getBoundingClientRect().top,hanzi:e.querySelector('.m13-hanzi').getBoundingClientRect().top})),
              ipaFont:getComputedStyle(document.querySelector('.m13-ipa')).fontFamily,
              loadedDoulos:[...document.fonts].filter(f=>f.family==='PTB-Doulos').map(f=>f.status),
              duplicateHeading:document.querySelectorAll('.mandarin-ipa-page h1').length,
              duplicateClose:document.querySelectorAll('.mandarin-ipa-page [aria-label="关闭模块"]').length,
              globalTransport:document.querySelectorAll('.global-transport').length,
              resources:performance.getEntriesByType('resource').map(e=>e.name),
              status:[...document.querySelectorAll('.statusbar span')].map(e=>e.textContent.trim()).filter(Boolean)
            }))()""",
            diagnostics,
        )

    def tick() -> None:
        nonlocal index, pending
        if pending:
            return
        if time.monotonic() - started > 45:
            finish(False, f"stage_timeout_{index + 1}")
            return
        pending = True

        def ready(value: object) -> None:
            nonlocal index, pending
            pending = False
            if not value:
                return
            action = stages[index][1]
            report["stages"].append(index + 1)
            index += 1
            if action == "__light__":
                window.view.grab().save(str(out / "qt-m13-light.png"))
            elif action == "__dark__":
                window.view.grab().save(str(out / "qt-m13-dark.png"))
            else:
                window.page.runJavaScript(action)
            if index == len(stages):
                timer.stop()
                QTimer.singleShot(250, lambda: finish(True))

        window.page.runJavaScript(stages[index][0], ready)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(120)
    app.exec()
    report_path = out / "report.json"
    print(report_path)
    if not report.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
