"""Fixed source entry for the Windows bounded document child."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptb_documents.web_child import main
main()
