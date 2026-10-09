"""Hold a process lease before Qt and preserve the cache for worker lifetime."""
import atexit
import os
from pathlib import Path
import sys
from ptb_desktop.startup_cache import cache_root, maintenance, lease

_ptb_cache = cache_root()
try:
    if Path(sys._MEIPASS).is_relative_to(_ptb_cache / 'apps'):
        with maintenance(_ptb_cache):
            _ptb_lease = lease(_ptb_cache)
        atexit.register(_ptb_lease.close)
except Exception:
    if os.environ.get('PTB_OWNED_BOOTSTRAP_LOG'):
        import traceback
        Path(os.environ['PTB_OWNED_BOOTSTRAP_LOG']).write_text(traceback.format_exc(),'utf8')
        raise SystemExit(1)
    raise
