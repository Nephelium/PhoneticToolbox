# P01 technical probe only; not the P12 release specification.
from pathlib import Path

root = Path(SPECPATH).resolve().parents[1]
analysis = Analysis(
    [str(root / 'desktop/experiments/host_probe.py')],
    pathex=[str(root / 'desktop/experiments')],
    binaries=[],
    datas=[(str(root / 'frontend/experiments/audio-viewport/dist'), 'web')],
    hiddenimports=[],
    hookspath=[],
    excludes=['PySide6', 'tkinter', 'numpy', 'pytest', 'pyaudiowpatch', 'psutil'],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, analysis.binaries, analysis.datas, [],
    name='PhoneticToolbox-P01', debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False,
)
