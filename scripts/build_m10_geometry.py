"""Rebuild only the reviewed bridge with the included VTL 2.4 source archive."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/validation/m10/native-build'
OUT.mkdir(parents=True,exist_ok=True)
SRC=OUT/'source';SRC.mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT/'resources/vocal_tract/sources/VTL2.4-API-source.zip') as archive:
    for name in archive.namelist():
        path=Path(name)
        if path.parent.as_posix()=='Developer/Sources' and path.suffix in ('.cpp','.h'):
            (SRC/path.name).write_bytes(archive.read(name))
# Display readout may retain sub-millimetre openings; the default acoustic
# function and all native tube calculations keep the upstream thresholds.
header=SRC/'VocalTract.h';s=header.read_text('utf-8')
old='bool considerTongue, Tube::Articulator &articulator, bool debug = false);'
assert s.count(old)==1;s=s.replace(old,old.replace('false);','false, bool geometric = false);'));header.write_text(s,encoding='utf-8')
source=SRC/'VocalTract.cpp';s=source.read_text('utf-8')
changes={
    'double *lowerProfile, bool considerTongue, Tube::Articulator &articulator, bool debug)':
    'double *lowerProfile, bool considerTongue, Tube::Articulator &articulator, bool debug, bool geometric)',
    'upperProfile[leftmost]-0.1':'upperProfile[leftmost]-(geometric ? 0.000001 : 0.1)',
    'upperProfile[rightmost]-0.1':'upperProfile[rightmost]-(geometric ? 0.000001 : 0.1)',
    'const double OPEN_THRESHOLD = 0.1;':'const double OPEN_THRESHOLD = geometric ? 0.000001 : 0.1;',
    'const double CUTOFF_THRESHOLD = 0.01;':'const double CUTOFF_THRESHOLD = geometric ? 0.000001 : 0.01;',
}
for a,b in changes.items():
    assert s.count(a)==1,a;s=s.replace(a,b)
source.write_text(s,encoding='utf-8')
vswhere=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
vs=subprocess.check_output([str(vswhere),'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip()
vcvars=Path(vs)/'VC/Auxiliary/Build/vcvars64.bat'
if not vcvars.is_file():raise RuntimeError('Existing MSVC compiler unavailable')
names='Geometry Surface VocalTract Tube Splines XmlNode XmlHelper Dsp Signal IirFilter Constants TlModel Matrix2x2'.split()
args=['/nologo','/LD','/O2','/EHsc','/std:c++14','/DWIN32','/D_USE_MATH_DEFINES','/MD',f'/I"{SRC}"',f'"{ROOT / "resources/vocal_tract/sources/geometry_bridge.cpp"}"']
args += [f'"{SRC/(n+".cpp")}"' for n in names]+['/Fegeometry_p2.dll']
rsp=OUT/'build.rsp';rsp.write_text('\n'.join(args),encoding='utf-8')
batch=OUT/'build.cmd';batch.write_text(f'@echo off\ncall "{vcvars}" >nul\ncl.exe @"{rsp}"\n',encoding='utf-8')
with (OUT/'build.log').open('w',encoding='utf-8') as log:
    result=subprocess.run(['cmd.exe','/d','/c',str(batch)],cwd=OUT,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:raise RuntimeError((OUT/'build.log').read_text('utf-8',errors='replace')[-3000:])
binary=OUT/'geometry_p2.dll'
(ROOT/'resources/vocal_tract/native/geometry_p2.dll').write_bytes(binary.read_bytes())
print(json.dumps({'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'compiler':str(vcvars),'source_archive_sha256':hashlib.sha256((ROOT/'resources/vocal_tract/sources/VTL2.4-API-source.zip').read_bytes()).hexdigest()}))
