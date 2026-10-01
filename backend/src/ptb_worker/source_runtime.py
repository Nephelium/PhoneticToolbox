"""Bind a source/snapshot bootstrap to its sibling core, without ambient PYTHONPATH."""
from pathlib import Path
import sys


def use_matching_core_source(bootstrap):
    worker=Path(bootstrap).resolve().parent
    # Installed wheels use their installed dependency. Source launch and frozen
    # source snapshots have this exact reviewed sibling layout.
    if worker.name!='ptb_worker' or worker.parent.name!='src' or worker.parent.parent.name!='backend':
        return
    core=worker.parents[2]/'packages/phonetic_core/src'
    if not (core/'phonetic_core/__init__.py').is_file():
        raise RuntimeError('Matching core source is missing from the application snapshot')
    sys.path.insert(0,str(core))
