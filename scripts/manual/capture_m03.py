"""Capture real M03 states in an owned, maximized Windows Qt workbench.

The authorised EGG recording is copied to a private run directory. The
recording, shared manual assets, project.json and other capture scripts are
never modified by this tool.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav")
for directory in ("desktop/src", "backend/src", "packages/phonetic_core/src", "scripts"):
    sys.path.insert(0, str(ROOT / directory))
os.environ["PYTHONPATH"] = os.pathsep.join(
    str(ROOT / directory)
    for directory in ("desktop/src", "backend/src", "packages/phonetic_core/src", "scripts")
)
os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS", "--mute-audio --autoplay-policy=no-user-gesture-required"
)
os.environ["PTB_EGG_PYTHON"] = str(ROOT / ".venv/m03-compatible/python.exe")


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def main() -> None:
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not (ROOT / "frontend/dist/index.html").is_file():
        raise FileNotFoundError(ROOT / "frontend/dist/index.html")
    if not (ROOT / ".venv/m03-compatible/python.exe").is_file():
        raise FileNotFoundError(ROOT / ".venv/m03-compatible/python.exe")

    out = ROOT / "output/manual-work/m03-captures" / uuid4().hex
    out.mkdir(parents=True)
    screenshots = out / "screenshots"
    screenshots.mkdir()
    inputs = out / "inputs"
    inputs.mkdir()
    saved = out / "saved"
    saved.mkdir()
    cache = out / "cache"
    cache.mkdir()
    initialize_local_files(cache)

    original_sha256 = digest(SOURCE)
    copied = inputs / "3.wav"
    shutil.copyfile(SOURCE, copied)
    if digest(copied) != original_sha256:
        raise AssertionError("The private input copy differs from the authorised recording")

    template = ROOT / "output/validation/p06/local-state.sqlite3"
    db = out / "jobs.sqlite3"
    with sqlite3.connect(template.as_uri() + "?mode=ro", uri=True) as source_db:
        with sqlite3.connect(db) as owned_db:
            if source_db.execute(
                "SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')"
            ).fetchone():
                raise RuntimeError("The read-only database template contains unfinished jobs")
            source_db.backup(owned_db)

    report: dict = {
        "success": False,
        "scope": "Actual Windows Qt workbench; owned hidden maximized window; physical DPI and playback unverified",
        "chapterId": "m03",
        "source": {
            "copiedFilename": copied.name,
            "sha256": original_sha256,
            "bytes": copied.stat().st_size,
            "originalUnchanged": None,
        },
        "out": str(out),
        "checks": [],
        "captures": [],
        "errors": [],
        "terminations": [],
    }
    choice = {"folder": inputs}
    QFileDialog.getExistingDirectory = lambda *args, **kwargs: str(choice["folder"])
    register_scheme()
    app = QApplication(["PTB-owned-M03-manual-capture"])
    app.setApplicationName("PTB-owned-M03-manual-capture")
    workbench = Workbench(
        ROOT / "frontend/dist",
        test=True,
        jobs_path=db,
        local_files_root=cache,
        vocal_profile=out / "vocal",
        reaper_binary=ROOT / "phonetic_toolbox/core/acoustic/reaper.exe",
    )
    workbench.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    workbench.showMaximized()
    workbench.page.renderProcessTerminated.connect(
        lambda status, code: report["terminations"].append([status.name, code])
    )

    def pause(milliseconds: int = 100) -> None:
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def js(code: str):
        loop = QEventLoop()
        values = []
        workbench.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
        QTimer.singleShot(6000, loop.quit)
        loop.exec()
        if not values:
            raise RuntimeError("JavaScript response timed out")
        return values[0]

    def until(code: str, seconds: int = 60) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code):
                return
            pause(120)
        detail = js("document.querySelector('.egg-page')?.innerText.slice(0,1200)")
        raise AssertionError(f"UI timeout: {code}; current M03: {detail}")

    def click(label: str) -> None:
        selected = js(
            "(()=>{const e=[...document.querySelectorAll('button')]"
            ".find(b=>b.offsetParent&&b.textContent.trim()==="
            + json.dumps(label, ensure_ascii=False)
            + ");if(!e||e.disabled)return false;e.click();return true})()"
        )
        if not selected:
            raise AssertionError(f"Button unavailable: {label}")
        pause(120)

    def snapshot(name: str, caption: str) -> None:
        if not workbench.isMaximized():
            raise AssertionError("M03 screenshot window is not maximized")
        until("document.documentElement.dataset.theme==='light'")
        pause(400)
        workbench.view.repaint()
        workbench.view.grab()
        pause(250)
        app.processEvents()
        pixmap = workbench.view.grab()
        file = screenshots / f"{name}.png"
        if not pixmap.save(str(file)):
            raise RuntimeError(f"Cannot save {file}")
        report["captures"].append(
            {
                "id": name,
                "chapterId": "m03",
                "file": str(file),
                "caption": caption,
                "distribution": "software-only",
                "maximized": workbench.isMaximized(),
                "window": [workbench.width(), workbench.height()],
                "frame": [
                    workbench.frameGeometry().width(),
                    workbench.frameGeometry().height(),
                ],
                "image": [pixmap.width(), pixmap.height()],
                "devicePixelRatio": pixmap.devicePixelRatio(),
                "theme": js("document.documentElement.dataset.theme"),
                "palette": js("document.documentElement.dataset.palette"),
                "fontSize": js("getComputedStyle(document.documentElement).fontSize"),
                "sha256": digest(file),
            }
        )
        print(f"Captured {name}", flush=True)

    def idle() -> None:
        until(
            "[...document.querySelectorAll('.egg-page button')]"
            ".some(b=>b.textContent.trim()==='保存 CSV / 三图'&&!b.disabled)"
            "&&document.querySelector('.egg-live-status')?.textContent.trim()==='实时预览'",
            120,
        )
        until(
            "document.querySelectorAll('.egg-four-plots .scientific-plot>svg').length===4"
        )

    try:
        until("!!document.querySelector('.home-page')&&document.fonts.status==='loaded'")
        if not workbench.isMaximized():
            raise AssertionError("Owned Qt workbench did not maximize")
        report["checks"].append("Owned Windows Qt workbench maximized before every capture")
        click("设置")
        click("浅色")
        until("document.documentElement.dataset.theme==='light'")
        click("EGG 信号分析")
        until("!!document.querySelector('.egg-page')")
        click("打开音频目录")
        until("document.querySelectorAll('.egg-source select option').length>=2")
        selected = js(
            "(()=>{const e=document.querySelector('.egg-source select');"
            "const o=[...e.options].find(x=>x.textContent.trim()==='3.wav');"
            "if(!o)return false;e.value=o.value;"
            "e.dispatchEvent(new Event('change',{bubbles:true}));return true})()"
        )
        if not selected:
            raise AssertionError("Authorised EGG file is absent from the workbench")
        idle()
        report["checks"].append("Authorised stereo recording copied and analysed in the live workbench")

        # The known natural vowel segment is 40.0–40.5 s in the unchanged recording.
        selected = js(
            "(()=>{const a=document.querySelectorAll('.egg-bottom-bar .selection-controls input');"
            "if(a.length!==2)return false;"
            "for(const [i,v] of [[1,40.5],[0,40]]){a[i].value=v;"
            "a[i].dispatchEvent(new Event('input',{bubbles:true}));"
            "a[i].dispatchEvent(new Event('change',{bubbles:true}));}return true})()"
        )
        if not selected:
            raise AssertionError("M03 bottom selection controls were unavailable")
        idle()
        if not js(
            "document.querySelector('.egg-current')?.innerText.includes('左：EGG · 右：音频')"
        ):
            raise AssertionError("The visible channel role differs from the chapter")
        snapshot(
            "m03-analysis-roi-light",
            "同步 EGG 与音频的 40.00–40.50 秒分析选区；CQ/SQ、语谱及音频和 EGG 微观图均为实时计算结果。",
        )

        enabled = js(
            "(()=>{const a=document.querySelectorAll('.egg-checks input');"
            "if(a.length!==3)return false;a.forEach(e=>{if(!e.checked)e.click()});return true})()"
        )
        if not enabled:
            raise AssertionError("The three independent F0 switches are unavailable")
        idle()
        until("document.querySelector('.spec-pane')?.innerText.includes('REAPER F0')")
        snapshot(
            "m03-three-f0-light",
            "同一选区开启 Praat、GCI 与 REAPER 三条独立 F0 来源；右轴、曲线和左栏开关同时可见。",
        )
        report["checks"].append("Real Praat, GCI and native REAPER F0 preview completed")

        click("批量分析")
        until("document.querySelector('dialog')?.innerText.includes('待处理文件')")
        snapshot(
            "m03-batch-parameters-light",
            "EGG 批量分析弹窗：高低通、GCI/GOI、声道交换、可选三图与三种 F0，以及文件全选和提交入口。",
        )
        click("返回工作台")
        until("!document.querySelector('dialog')")
        report["checks"].append("Independent batch parameter dialog shown without submitting a batch")

        js(
            "document.querySelectorAll('.egg-checks input')"
            ".forEach(e=>{if(e.checked)e.click()})"
        )
        idle()
        click("逆滤波 IF")
        until(
            "document.querySelectorAll('.inverse-grid .scientific-plot>svg').length===4",
            180,
        )
        until("document.querySelectorAll('dialog .audio-transport').length===2")
        if not js(
            "document.querySelector('dialog')?.innerText.includes('固定 3 ms 窗')"
        ):
            raise AssertionError("IF diagnostics did not show the fixed-window statement")
        snapshot(
            "m03-inverse-result-light",
            "40.00–40.50 秒真实片段的简化逆滤波结果；图组、LP 设置和固定 3 ms 取窗提示均来自同次任务。",
        )
        choice["folder"] = saved
        click("选择目录保存完整结果")
        deadline = time.monotonic() + 60
        while not list(saved.glob("*.ptb.json")) and time.monotonic() < deadline:
            pause(200)
        if not list(saved.glob("*.ptb.json")):
            raise AssertionError("IF result did not save a provenance JSON in the owned directory")
        until("document.querySelector('dialog')?.innerText.includes('已保存')", 30)
        snapshot(
            "m03-inverse-saved-light",
            "同一逆滤波任务选择独立目录后显示保存反馈；结果包含归一化分析音频、IF 估计与来源记录。",
        )
        report["savedFiles"] = [
            {"name": file.name, "sha256": digest(file), "bytes": file.stat().st_size}
            for file in sorted(saved.iterdir())
            if file.is_file()
        ]
        report["checks"].append("IF result saved to this run's directory; output files hashed")
        report["success"] = True
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        try:
            diagnostic = screenshots / "diagnostic-failure.png"
            workbench.view.grab().save(str(diagnostic))
            report["diagnostic"] = str(diagnostic)
        except Exception:
            pass
        raise
    finally:
        report["source"]["originalUnchanged"] = digest(SOURCE) == original_sha256
        if not report["source"]["originalUnchanged"]:
            report["success"] = False
            report["errors"].append(
                {"type": "SourceChanged", "message": "Authorised original SHA-256 changed"}
            )
        if report["terminations"]:
            report["success"] = False
        (out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "success": report["success"],
                    "report": str(out / "report.json"),
                    "captures": len(report["captures"]),
                    "errors": report["errors"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        workbench.closing = True
        workbench.close()
        pause(300)
        app.quit()


if __name__ == "__main__":
    main()
