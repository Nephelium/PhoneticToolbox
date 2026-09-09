"""P01 native availability evidence, deliberately separate from scientific parity."""
import ctypes as C
import hashlib
import json
import math
import statistics
import struct
import subprocess
import wave
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'output/validation/p01/native'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    resource = ROOT / 'phonetic_toolbox/resources/vocal_tract'
    source_zip = resource / 'sources/VTL2.4-API-source.zip'
    with zipfile.ZipFile(source_zip) as archive:
        inventory = archive.namelist()
        cpp = [name for name in inventory if name.endswith(('.cpp', '.h'))]
        projects = [name for name in inventory if name.endswith(('.vcxproj', 'CMakeLists.txt', 'Makefile'))]
    lib = C.CDLL(str(resource / 'native/VocalTractLabApi.dll'))
    signatures = {'vtlGetVersion': ([C.c_char_p], None), 'vtlInitialize': ([C.c_char_p], C.c_int),
                  'vtlGetConstants': ([C.POINTER(C.c_int)] * 4, C.c_int), 'vtlClose': ([], C.c_int)}
    for name, (args, result) in signatures.items():
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = result
    version = C.create_string_buffer(256)
    lib.vtlGetVersion(version)
    initialized = lib.vtlInitialize(str(resource / 'native/JD2.speaker').encode('utf-8'))
    assert initialized == 0, initialized
    try:
        constants = [C.c_int() for _ in range(4)]
        status = lib.vtlGetConstants(*[C.byref(value) for value in constants])
        assert status == 0 and constants[0].value > 0
    finally:
        closed = lib.vtlClose()
    fixture = OUTPUT / 'known-440hz.wav'
    with wave.open(str(fixture), 'wb') as audio:
        audio.setparams((1, 2, 44100, 88200, 'NONE', 'not compressed'))
        audio.writeframes(b''.join(struct.pack('<h', round(0.4 * 32767 * math.sin(2 * math.pi * 440 * i / 44100))) for i in range(88200)))
    reaper = ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe'
    command = [str(reaper), '-i', str(fixture), '-f', str(OUTPUT / 'probe.f0'), '-p', str(OUTPUT / 'probe.pm'), '-a']
    result = subprocess.run(command, cwd=OUTPUT, capture_output=True, timeout=30)
    (OUTPUT / 'reaper-output.txt').write_bytes(result.stdout + result.stderr)
    assert result.returncode == 0, result.returncode
    lines = (OUTPUT / 'probe.f0').read_text('utf-8').split('EST_Header_End', 1)[1].splitlines()
    voiced = [float(parts[2]) for line in lines if len(parts := line.split()) >= 3 and float(parts[2]) > 0]
    assert voiced, 'REAPER produced no voiced estimates'
    report = {
        'VTL': {'api_version_string': version.value.decode(), 'initialize_exit': initialized, 'close_exit': closed,
                'constants': dict(zip(['sample_rate_hz', 'tube_sections', 'tract_parameters', 'glottis_parameters'], [x.value for x in constants])),
                'dll_sha256': sha(resource / 'native/VocalTractLabApi.dll'), 'source_zip_sha256': sha(source_zip),
                'source_files': len(cpp), 'build_project_files': projects,
                'source_relation': 'retained official 2.4 API source archive; binary reproducibility not yet established',
                'linux_macos_build': 'not executed; current upstream backend is not proof of compatibility with the 2.4 binary'} ,
        'REAPER': {'exit_code': result.returncode, 'binary_sha256': sha(reaper), 'input_hz': 440,
                   'voiced_frames': len(voiced), 'median_f0_hz': statistics.median(voiced),
                   'source': 'https://github.com/google/REAPER', 'license': 'Apache-2.0',
                   'included_commit': None, 'linux_macos_build': 'not executed'},
        'scope': 'Windows native availability probe; not full acoustic validation or cross-platform acceptance',
    }
    (OUTPUT / 'native-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
