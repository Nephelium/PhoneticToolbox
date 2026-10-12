"""Frozen feedback inbox: deliberately no scientific runtime imports."""
from pathlib import Path
import os
import sys

def main():
    from ptb_desktop.feedback_manager import run
    root=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[3]))
    return run(root/'frontend')

if __name__ == '__main__':
    owned_log=os.environ.get('PTB_OWNED_BOOTSTRAP_LOG')
    if owned_log:
        try:
            result=main()
        except Exception:
            import traceback
            Path(owned_log).write_text(traceback.format_exc(),encoding='utf-8')
            result=1
        raise SystemExit(result)
    raise SystemExit(main())
