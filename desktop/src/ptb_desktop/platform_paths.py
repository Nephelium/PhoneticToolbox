"""Application-owned user data, independent of the working/build directory."""
import os
from pathlib import Path
import sys


def user_data_root(*, platform=None, environ=None, home=None):
    platform=sys.platform if platform is None else platform
    environ=os.environ if environ is None else environ
    home=Path.home() if home is None else Path(home)
    if platform=='win32':
        candidate=Path(environ.get('LOCALAPPDATA',''))
        base=candidate if candidate.is_absolute() else home/'AppData/Local'
    elif platform=='darwin':base=home/'Library/Application Support'
    elif platform=='linux':
        candidate=Path(environ.get('XDG_DATA_HOME',''))
        base=candidate if candidate.is_absolute() else home/'.local/share'
    else:raise ValueError('Unsupported desktop platform')
    return base/'PhoneticToolbox/v3'
