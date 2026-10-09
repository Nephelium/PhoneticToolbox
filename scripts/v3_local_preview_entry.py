"""Local trial entry. Bundled source snapshot; optional runtimes stay separate."""
import json
import os
from pathlib import Path
import sys


def configure(*, runtime=True):
    bundle = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    # External scientific bootstraps require real files in the reviewed layout.
    for relative in ('packages/phonetic_core/src', 'desktop/src', 'backend/src'):
        path = bundle / relative
        if path.is_dir():
            sys.path.insert(0, str(path))
    if not getattr(sys, 'frozen', False):
        # The source verifier's child service must use this checkout as well.
        os.environ['PYTHONPATH']=os.pathsep.join(str(bundle/p) for p in
            ('packages/phonetic_core/src','desktop/src','backend/src'))
        for key,relative in (
            ('PTB_EGG_PYTHON','.venv/m03-compatible/python.exe'),
            ('PTB_M05_PYTHON','.venv/m05/Scripts/python.exe'),
            ('PTB_M11_COMPONENT_ROOT','output/m11c-028b881d')):
            if (bundle/relative).exists():os.environ.setdefault(key,str(bundle/relative))
    config = bundle / 'local-preview.json'
    portable = bundle / 'desktop-bundle.json'
    if not runtime:
        return bundle
    if portable.is_file():
        from ptb_desktop.compact_runtime import expand
        expand(bundle)
        from ptb_desktop.bundle_manifest import runtime_bindings
        # A portable build is pinned to its own validated runtime paths.
        os.environ.update(runtime_bindings(bundle))
    elif config.is_file():
        value = json.loads(config.read_text(encoding='utf-8'))
        if value.get('portable') is not False:
            raise ValueError('External runtime preview cannot be marked portable')
        for key in ('PTB_EGG_PYTHON', 'PTB_M05_PYTHON', 'PTB_M11_COMPONENT_ROOT'):
            path = value.get('runtimes', {}).get(key)
            if path and Path(path).exists():
                os.environ.setdefault(key, path)
    return bundle


def main():
    helper = len(sys.argv) > 1 and sys.argv[1] == '--ptb-apply-update'
    bundle = configure(runtime=not helper)
    if sys.argv[1:]==['--ptb-clear-user-caches']:
        from ptb_desktop.cache_cleanup import result_caches
        return 0 if result_caches()['complete'] else 21
    if helper:
        from ptb_desktop.update_apply import helper_main
        return helper_main(sys.argv[2:])
    if len(sys.argv) == 3 and sys.argv[1] == '--verify-distribution':
        from m10_recording_entry import restore_worker_pipes
        restore_worker_pipes()
        from verify_distribution import verify
        return verify(bundle, Path(sys.argv[2]).absolute())
    if len(sys.argv) == 4 and sys.argv[1] == '--verify-presentation-preview':
        from m10_recording_entry import restore_worker_pipes
        restore_worker_pipes()
        from verify_p19_r14_presentation import verify
        return verify(bundle, Path(sys.argv[2]).absolute(), sys.argv[3])
    if len(sys.argv) == 5 and sys.argv[1] == '--verify-maximize-preview':
        from m10_recording_entry import restore_worker_pipes
        restore_worker_pipes()
        from verify_p19_r18_presentation import verify
        return verify(bundle, Path(sys.argv[2]).absolute(), sys.argv[3], sys.argv[4])
    if len(sys.argv) == 3 and sys.argv[1] == '--verify-m13-preview':
        from m10_recording_entry import restore_worker_pipes
        restore_worker_pipes()
        from verify_m13_package import verify
        return verify(bundle, Path(sys.argv[2]).absolute())
    if len(sys.argv) == 3 and sys.argv[1] == '--verify-m11-preview':
        from m10_recording_entry import restore_worker_pipes
        restore_worker_pipes()
        from verify_m11_r2_qt import verify
        registry=Path(os.environ['PTB_M11_COMPONENT_ROOT'])/'registry.json'
        return verify(bundle,Path(sys.argv[2]).absolute(),registry=registry)
    if len(sys.argv) == 3 and sys.argv[1] == '--verify-preview':
        from m10_recording_entry import restore_worker_pipes
        restore_worker_pipes()
        from verify_v3_local_preview import verify
        return verify(bundle, Path(sys.argv[2]).absolute())
    if len(sys.argv) == 4 and sys.argv[1] == '--verify-natural-preview':
        from m10_recording_entry import restore_worker_pipes
        restore_worker_pipes()
        from verify_v3_local_preview import verify
        return verify(bundle, Path(sys.argv[2]).absolute(), Path(sys.argv[3]).absolute())
    # Keep all fixed worker dispatch and unknown-switch rejection in one entry.
    from research_entry import main as research_main
    if len(sys.argv) == 1:
        from ptb_desktop.platform_paths import user_data_root
        sys.argv += ['--local-root', str(user_data_root() / 'local-preview-20260927')]
    return research_main()


def guarded_main():
    # Owned distribution validation must leave a diagnostic rather than a
    # PyInstaller popup, including failures before the validator imports.
    if len(sys.argv) == 3 and sys.argv[1] == '--verify-distribution':
        try:
            return main()
        except Exception:
            import traceback
            output = Path(sys.argv[2]).absolute()
            output.mkdir(parents=True, exist_ok=True)
            (output/'startup-error.log').write_text(traceback.format_exc(),encoding='utf-8')
            return 1
    return main()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(guarded_main())
