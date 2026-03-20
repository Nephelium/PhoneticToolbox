# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

CODEC_DLLS_TO_PRUNE = {
    "avcodec-62.dll",
    "avdevice-62.dll",
    "avfilter-11.dll",
    "avformat-62.dll",
    "avutil-60.dll",
    "swresample-6.dll",
    "swscale-9.dll",
    "postproc-59.dll",
    "libx265.dll",
    "libx264-164.dll",
    "aom.dll",
    "svtav1enc.dll",
}

FFMPEG_TOOL_FILES = {
    "ffmpeg.exe",
    "avcodec-62.dll",
    "avdevice-62.dll",
    "avfilter-11.dll",
    "avformat-62.dll",
    "avutil-60.dll",
    "swresample-6.dll",
    "swscale-9.dll",
    "postproc-59.dll",
    "libx265.dll",
    "libx264-164.dll",
    "aom.dll",
    "svtav1enc.dll",
}

def dedupe_pairs(pairs):
    seen = set()
    result = []
    for src, dst in pairs:
        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)
        result.append((src, dst))
    return result

def keep_binary(pair):
    src, _ = pair
    normalized = src.lower().replace("/", "\\")
    name = Path(src).name.lower()
    if "\\site-packages\\jaxlib\\" in normalized:
        return False
    if "\\site-packages\\jax\\" in normalized:
        return False
    if "\\site-packages\\llvmlite\\" in normalized:
        return False
    if "\\site-packages\\numba\\" in normalized:
        return False
    if (
        "kaldi-" in name
        or "fst" in name
        or "ngram" in name
        or "pynini" in name
        or "baumwelch" in name
    ):
        return False
    if "\\library\\bin\\" in normalized and name in CODEC_DLLS_TO_PRUNE:
        return False
    return True

def keep_collected_binary(entry):
    if len(entry) < 2:
        return True
    name = str(entry[0]).lower()
    src = str(entry[1]).lower().replace("/", "\\")
    if name.startswith("tools\\"):
        return True
    if "\\site-packages\\pyqt6\\qt6\\bin\\" in src and name in {
        "pyqt6\\qt6\\bin\\msvcp140.dll",
        "pyqt6\\qt6\\bin\\msvcp140_1.dll",
        "pyqt6\\qt6\\bin\\vcruntime140.dll",
        "pyqt6\\qt6\\bin\\vcruntime140_1.dll",
        "pyqt6\\qt6\\bin\\concrt140.dll",
    }:
        return False
    if "\\site-packages\\jaxlib\\" in src or "\\site-packages\\jax\\" in src:
        return False
    if "\\site-packages\\llvmlite\\" in src or "\\site-packages\\numba\\" in src:
        return False
    if any(token in name for token in ["kaldi-", "fst", "ngram", "pynini", "baumwelch"]):
        return False
    src_basename = Path(src).name.lower()
    if "\\library\\bin\\" in src and src_basename in CODEC_DLLS_TO_PRUNE:
        return False
    return True

binaries = []
binaries += [(str(path), ".") for path in (Path(sys.prefix) / "Library" / "bin").glob("*.dll")]
binaries += [(str(path), ".") for path in (Path(sys.prefix) / "DLLs").glob("*.dll")]
for name in [
    "MSVCP140.dll",
    "MSVCP140_1.dll",
    "VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll",
    "CONCRT140.dll",
]:
    candidate = Path(sys.prefix) / "Library" / "bin" / name
    if candidate.exists():
        binaries += [(str(candidate), ".")]
binaries += collect_dynamic_libs("scipy")
binaries += collect_dynamic_libs("numpy")
def keep_mediapipe_data(src):
    normalized = src.lower().replace("\\", "/")
    if normalized.endswith((".dll", ".pyd")):
        return False
    return any(
        marker in normalized
        for marker in (
            "/mediapipe/modules/face_landmark/",
            "/mediapipe/modules/face_detection/",
            "/mediapipe/modules/face_geometry/",
            "/mediapipe/modules/iris_landmark/",
        )
    )

mediapipe_datas = [
    (src, dst)
    for src, dst in collect_data_files("mediapipe")
    if keep_mediapipe_data(src)
]
mediapipe_binaries = collect_dynamic_libs("mediapipe")
mediapipe_hiddenimports = [
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.face_mesh",
    "mediapipe.python.solutions.face_mesh_connections",
    "mediapipe.python.solutions.drawing_utils",
    "mediapipe.python.solutions.drawing_styles",
] + collect_submodules("mediapipe.python._framework_bindings")
binaries += mediapipe_binaries
binaries = dedupe_pairs(binaries)
binaries = [pair for pair in binaries if keep_binary(pair)]
ffmpeg_tool_datas = []
for name in FFMPEG_TOOL_FILES:
    candidate = Path(sys.prefix) / "Library" / "bin" / name
    if candidate.exists():
        ffmpeg_tool_datas.append((str(candidate), "tools"))
datas = dedupe_pairs(
    [
        ('D:\\PhoneticToolbox\\PhoneticToolbox_v2\\PhoneticToolbox.ico', '.'),
        ('D:\\PhoneticToolbox\\PhoneticToolbox_v2\\Phonetic_Export', 'Phonetic_Export'),
        ('D:\\PhoneticToolbox\\PhoneticToolbox_v2\\phonetic_toolbox\\gui\\resources\\ipa_trans', 'phonetic_toolbox\\gui\\resources\\ipa_trans'),
    ] + mediapipe_datas + ffmpeg_tool_datas
)


a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=mediapipe_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'auto_alignment',
        'pytest',
        '_pytest',
        'hypothesis',
        'coverage',
        'jax',
        'jaxlib',
        'numba',
        'llvmlite',
        'kalpy',
        '_kalpy',
        'pynini',
        '_pynini',
        'pywrapfst',
        '_pywrapfst',
        'montreal_forced_aligner',
        'mediapipe.examples',
        'mediapipe.tasks.python.test',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
filtered_binaries = [entry for entry in a.binaries if keep_collected_binary(entry)]

exe = EXE(
    pyz,
    a.scripts,
    filtered_binaries,
    a.datas,
    [],
    name='PhoneticToolbox',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['D:\\PhoneticToolbox\\PhoneticToolbox_v2\\PhoneticToolbox.ico'],
)
