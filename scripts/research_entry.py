"""Unified local research artifact with explicit frozen worker dispatch."""
import os
from pathlib import Path
import sys


def main():
    # Child selection happens before importing Qt or constructing any window.
    from m10_recording_entry import restore_worker_pipes
    if len(sys.argv) > 1 and sys.argv[1] == '--ptb-worker':
        restore_worker_pipes()
        from ptb_worker.process_entry import MODULES, dispatch
        if len(sys.argv) < 3 or sys.argv[2] not in MODULES: return 2
        dispatch(sys.argv[2], sys.argv[3:]); return 0
    if '--local-service' in sys.argv:
        restore_worker_pipes(); sys.argv.remove('--local-service')
        from ptb_api.cli import main as service
        service(); return 0
    if '--m10-worker' in sys.argv:
        restore_worker_pipes()
        from ptb_desktop.vocal_tract.worker import main as worker
        worker(); return 0
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root', type=Path)
    parser.add_argument('--verify-repair', type=Path)
    parser.add_argument('--verify-m12', type=Path)
    parser.add_argument('--verify-m12-r1', type=Path)
    parser.add_argument('--verify-m12-r2', type=Path)
    parser.add_argument('--verify-m12-r3', type=Path)
    args = parser.parse_args()  # Unknown switches fail closed, never open a GUI.
    bundle = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    from ptb_desktop.platform_paths import user_data_root
    root = args.local_root or user_data_root() / 'research-v1'
    from ptb_worker.local_workspace import prepare_workspace
    database, files = prepare_workspace(root, bundle / 'backend/migrations')
    reaper = bundle / 'resources/research' / ('reaper.exe' if sys.platform=='win32' else 'reaper')
    if not getattr(sys, 'frozen', False) and sys.platform=='win32':
        reaper = bundle / 'phonetic_toolbox/core/acoustic/reaper.exe'
    if args.verify_repair:
        restore_worker_pipes()
        from verify_research_repair import verify
        return verify(bundle, database, files, reaper, args.verify_repair)
    if args.verify_m12 or args.verify_m12_r1 or args.verify_m12_r2 or args.verify_m12_r3:
        restore_worker_pipes()
        from verify_m12_qt import main as verify
        verify(bundle=bundle, out=args.verify_m12 or args.verify_m12_r1 or args.verify_m12_r2 or args.verify_m12_r3, database=database, cache=files,
               reaper=reaper, r1=bool(args.verify_m12_r1),r2=bool(args.verify_m12_r2),r3=bool(args.verify_m12_r3))
        return 0
    from ptb_desktop.host import run
    return run(bundle / 'frontend/dist', jobs_path=database, local_files_root=files,
               reaper_binary=reaper, vocal_resources=bundle / 'resources/vocal_tract/native')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
