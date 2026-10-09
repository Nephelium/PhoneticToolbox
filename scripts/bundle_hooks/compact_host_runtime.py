"""Run before the standard Qt hooks. No application or devices are opened."""
import sys
import os
from ptb_desktop.compact_host import expand

try:
    expand(sys._MEIPASS)
except Exception:
    # Owned automation can leave the complete failure without a bootloader
    # dialog. It still exits nonzero, so launch/upgrade verification must fail.
    if os.environ.get('PTB_OWNED_BOOTSTRAP_LOG'):
        import traceback
        from pathlib import Path
        try:
            Path(os.environ['PTB_OWNED_BOOTSTRAP_LOG']).write_text(traceback.format_exc(),encoding='utf8')
        finally:
            raise SystemExit(1)
    raise
