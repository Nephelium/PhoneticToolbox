"""Bind a source/snapshot bootstrap to its sibling core, without ambient PYTHONPATH."""
from pathlib import Path
import sys


def prepare_native_import_paths():
    """Extend only native DLL load paths, preserving Python resource paths.

    MediaPipe's graph paths contain forward slashes and cannot use the Win32
    extended namespace. Ordinary Python module paths must remain unchanged.
    """
    if sys.platform != 'win32':
        return
    def ordinary(value):
        if value.startswith('\\\\?\\UNC\\'):
            return '\\\\' + value[8:]
        return value.removeprefix('\\\\?\\')
    # CreateProcess needs the extended executable name beyond MAX_PATH. Undo
    # that spelling for Python resources before third-party packages import.
    sys.path[:] = [ordinary(p) for p in sys.path]
    for name in ('prefix', 'exec_prefix', 'base_prefix', 'base_exec_prefix'):
        setattr(sys, name, ordinary(getattr(sys, name)))
    sys.path_importer_cache.clear()
    from importlib.machinery import ExtensionFileLoader, PathFinder
    if any(getattr(finder, '_ptb_long_native_paths', False) for finder in sys.meta_path):
        return
    def extended(value):
        path = str(Path(value).absolute())
        if path.startswith('\\\\?\\'):
            return path
        if path.startswith('\\\\'):
            return '\\\\?\\UNC\\' + path[2:]
        return '\\\\?\\' + path
    class NativeLoader(ExtensionFileLoader):
        def create_module(self, spec):
            original = spec.origin
            try:
                spec.origin = extended(original)
                module = super().create_module(spec)
            finally:
                spec.origin = original
            if module is not None:
                module.__file__ = original
            return module

    class NativeFinder:
        _ptb_long_native_paths = True
        dll_directories = []

        @staticmethod
        def find_spec(fullname, path=None, target=None):
            spec = PathFinder.find_spec(fullname, path, target)
            if spec is not None and isinstance(spec.loader, ExtensionFileLoader):
                spec.loader = NativeLoader(fullname, spec.origin)
                return spec
            return None

    import os
    for relative in ('Library/bin', 'DLLs'):
        directory = Path(sys.prefix)/relative
        if directory.is_dir():
            NativeFinder.dll_directories.append(os.add_dll_directory(extended(str(directory))))
    sys.meta_path.insert(sys.meta_path.index(PathFinder), NativeFinder)


def use_matching_core_source(bootstrap):
    prepare_native_import_paths()
    worker=Path(bootstrap).resolve().parent
    # Installed wheels use their installed dependency. Source launch and frozen
    # source snapshots have this exact reviewed sibling layout.
    if worker.name!='ptb_worker' or worker.parent.name!='src' or worker.parent.parent.name!='backend':
        return
    core=worker.parents[2]/'packages/phonetic_core/src'
    if not (core/'phonetic_core/__init__.py').is_file():
        raise RuntimeError('Matching core source is missing from the application snapshot')
    sys.path.insert(0,str(core))
