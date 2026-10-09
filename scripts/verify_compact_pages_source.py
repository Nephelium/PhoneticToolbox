"""Own source-side setup for the compact client probe, before freezing it."""
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
for relative in ('desktop/src', 'backend/src', 'packages/phonetic_core/src'):
    sys.path.insert(0, str(ROOT / relative))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / relative) for relative in ('desktop/src', 'backend/src', 'packages/phonetic_core/src'))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'output/validation/compact-pages-source' / uuid4().hex
    output.mkdir(parents=True)
    profile = output / 'profile'; profile.mkdir()
    os.environ['LOCALAPPDATA'] = str(profile)
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio --disable-gpu --autoplay-policy=no-user-gesture-required'
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    from verify_compact_pages import verify
    database, cache = prepare_workspace(output / 'state', ROOT / 'backend/migrations')
    register_scheme(); app = QApplication(['compact-pages-owned-QA'])
    window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=database, local_files_root=cache,
                       vocal_resources=ROOT / 'resources/vocal_tract/native', vocal_profile=output / 'vocal')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen); window.showMaximized()
    report = dict(success=False, checks=[])
    try:
        loop = QEventLoop(); QTimer.singleShot(4000, loop.quit); loop.exec()
        if args.recording:
            from verify_m16_m17_frozen import verify as recording
            recording(window, output, report)
        verify(window, output, report)
        report['success'] = True
    except Exception:
        import traceback
        report['error'] = traceback.format_exc()
    finally:
        window.closing = True; window.close(); app.quit()
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf8')
        print(json.dumps(dict(success=report['success'], report=str(output / 'report.json'))), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
