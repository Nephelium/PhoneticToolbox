"""Fixed LPC child bootstrap; isolates scientific DLLs from the host process."""
import os
from pathlib import Path
import sys

prefix = Path(sys.prefix)
os.environ['PATH'] = os.pathsep.join([str(prefix),str(prefix/'Library/bin'),
    str(prefix/'Scripts'),str(Path(os.environ['SystemRoot'])/'System32'),os.environ['SystemRoot']])
os.environ['MPLCONFIGDIR'] = str(prefix/'var/cache/ptb-m03-matplotlib')
dll_handle = os.add_dll_directory(str(prefix/'Library/bin'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ptb_worker.source_runtime import use_matching_core_source
use_matching_core_source(__file__)
from ptb_worker.lpc_child import run
run()
