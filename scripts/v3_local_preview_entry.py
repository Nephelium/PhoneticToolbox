"""Local trial entry. Bundled source snapshot; optional runtimes stay separate."""
import json
import os
from pathlib import Path
import sys


def configure():
    bundle = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    # External scientific bootstraps require real files in the reviewed layout.
    for relative in ('packages/phonetic_core/src', 'desktop/src', 'backend/src'):
        path = bundle / relative
        if path.is_dir():
            sys.path.insert(0, str(path))
    config = bundle / 'local-preview.json'
    portable = bundle / 'desktop-bundle.json'
    if portable.is_file():
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
    bundle = configure()
    if len(sys.argv) == 3 and sys.argv[1] == '--verify-preview':
        from m10_recording_entry import restore_worker_pipes
        restore_worker_pipes()
        from verify_v3_local_preview import verify
        return verify(bundle, Path(sys.argv[2]).absolute())
    # Keep all fixed worker dispatch and unknown-switch rejection in one entry.
    from research_entry import main as research_main
    if len(sys.argv) == 1:
        from ptb_desktop.platform_paths import user_data_root
        sys.argv += ['--local-root', str(user_data_root() / 'local-preview-20260927')]
    return research_main()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
