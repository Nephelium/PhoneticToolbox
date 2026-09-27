"""Explicit main-package boundary; optional MFA prefixes are never collected."""
from pathlib import Path

def arguments(root):
    root=Path(root)
    args=['--add-data',str(root/'backend/src/ptb_worker/mfa/child.py')+';resources/mfa']
    for name in ('montreal_forced_aligner','_kalpy','kalpy','kaldi'):
        args+=['--exclude-module',name]
    return args
