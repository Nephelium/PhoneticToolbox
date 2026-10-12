"""Current-checkout entry for the standalone feedback administrator."""
from pathlib import Path
import sys
from workbench_source import bind_sources

ROOT = Path(__file__).resolve().parents[1]

if __name__=='__main__':
    bind_sources(ROOT)
    from ptb_desktop.feedback_manager import run
    raise SystemExit(run(ROOT/'frontend/dist'))
