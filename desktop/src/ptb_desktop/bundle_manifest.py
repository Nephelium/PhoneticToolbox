"""Resolve explicitly bundled runtimes without paths from a developer machine.

This checks the runtime binding, not scientific equivalence or code signing.
Local preview manifests remain separate and must never be labelled portable.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys

RUNTIMES=frozenset({'PTB_EGG_PYTHON','PTB_M05_PYTHON'})


def architecture(value):
    value=value.lower()
    return {'amd64':'x86_64','aarch64':'arm64'}.get(value,value)


def bundled_path(root,relative):
    if not isinstance(relative,str) or '\\' in relative or ':' in relative:
        raise ValueError('Invalid bundled path')
    p=PurePosixPath(relative)
    if p.is_absolute() or not p.parts or any(part in ('.','..') for part in p.parts):
        raise ValueError('Invalid bundled path')
    path=root.joinpath(*p.parts)
    if any(item.is_symlink() for item in (path,*path.parents) if item!=root.parent):
        raise ValueError('Bundled paths must not traverse symbolic links')
    if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Bundle path escape')
    return path


def runtime_bindings(root, *, host_platform=None, host_arch=None):
    root=Path(root).resolve()
    manifest=root/'desktop-bundle.json'
    if manifest.stat().st_size>1_000_000:raise ValueError('Bundle manifest too large')
    value=json.loads(manifest.read_text('utf-8'))
    target=sys.platform if host_platform is None else host_platform
    arch=architecture(platform.machine() if host_arch is None else host_arch)
    if (value.get('schema')!='desktop-bundle/1' or value.get('portable') is not True or
        value.get('platform')!=target or value.get('architecture')!=arch):
        raise ValueError('Bundle platform, architecture or portability mismatch')
    entries=value.get('runtimes',{})
    if not isinstance(entries,dict) or set(entries)-RUNTIMES:raise ValueError('Unsupported bundled runtime')
    bindings={}
    for key,item in entries.items():
        if not isinstance(item,dict):raise ValueError('Invalid bundled runtime')
        path=bundled_path(root,item.get('path'))
        if not path.is_file():raise ValueError('Bundled runtime missing')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=item.get('sha256'):raise ValueError('Bundled runtime hash mismatch')
        bindings[key]=str(path)
    return bindings
