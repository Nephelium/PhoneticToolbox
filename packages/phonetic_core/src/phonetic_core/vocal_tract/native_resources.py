"""Native ABI resource selection; finding a file does not certify its platform parity."""
from pathlib import Path
import platform as host
import sys


def _windows_process_architecture():
    """Query Windows when sanitized launchers omit processor environment fields."""
    import ctypes
    from ctypes import wintypes
    class SystemInfo(ctypes.Structure):
        _fields_ = [('architecture', wintypes.WORD), ('reserved', wintypes.WORD),
                    ('page_size', wintypes.DWORD), ('minimum_address', ctypes.c_void_p),
                    ('maximum_address', ctypes.c_void_p), ('processor_mask', ctypes.c_size_t),
                    ('processor_count', wintypes.DWORD), ('processor_type', wintypes.DWORD),
                    ('allocation_granularity', wintypes.DWORD), ('processor_level', wintypes.WORD),
                    ('processor_revision', wintypes.WORD)]
    info = SystemInfo()
    query = ctypes.WinDLL('kernel32').GetSystemInfo
    query.argtypes = [ctypes.POINTER(SystemInfo)]
    query.restype = None
    query(ctypes.byref(info))
    return {9: 'x86_64', 12: 'arm64'}.get(info.architecture, '')


def resolve_libraries(root, *, platform=None, arch=None):
    platform=sys.platform if platform is None else platform
    detected=host.machine() if arch is None else arch
    if arch is None and not detected and platform=='win32' and sys.platform=='win32':
        detected=_windows_process_architecture()
    arch=detected.lower()
    arch={'amd64':'x86_64','aarch64':'arm64'}.get(arch,arch)
    names={
        'win32':('VocalTractLabApi.dll','VocalTractLabAnalysis.dll','geometry_p2.dll'),
        'darwin':('libVocalTractLabApi.dylib','libVocalTractLabAnalysis.dylib','libgeometry_p2.dylib'),
        'linux':('libVocalTractLabApi.so','libVocalTractLabAnalysis.so','libgeometry_p2.so'),
    }
    if platform not in names or arch not in ('x86_64','arm64'):
        raise RuntimeError('native_platform_unavailable')
    root=Path(root)
    directory=root/f'{platform}-{arch}'
    # The legacy flat directory is the already-validated Windows x86_64 layout.
    if platform=='win32' and arch=='x86_64' and not directory.exists():directory=root
    result=tuple(directory/name for name in names[platform])
    if any(not path.is_file() or path.is_symlink() for path in result):
        raise RuntimeError('native_resource_unavailable')
    return result
