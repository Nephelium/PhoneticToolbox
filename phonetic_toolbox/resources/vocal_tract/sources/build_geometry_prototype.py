"""Build only the local geometry bridge with an existing MSVC installation."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'vendor/VTL2.4/API/Developer/Sources'
OUT=ROOT/'output/native'
OUT.mkdir(parents=True,exist_ok=True)
vswhere=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
vs=subprocess.check_output([str(vswhere),'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip()
vcvars=Path(vs)/'VC/Auxiliary/Build/vcvars64.bat'
assert vcvars.is_file(), 'Existing MSVC not found'
names='Geometry Surface VocalTract Tube Splines XmlNode XmlHelper Dsp Signal IirFilter Constants TlModel Matrix2x2'.split()
args=['/nologo','/LD','/O2','/EHsc','/std:c++14','/DWIN32','/D_USE_MATH_DEFINES','/MD',f'/I"{SOURCE}"',f'"{ROOT / "native/geometry_bridge.cpp"}"']
args += [f'"{SOURCE / (n+".cpp")}"' for n in names]
args += ['/Fegeometry_p2.dll']
rsp=OUT/'build.rsp'
rsp.write_text('\n'.join(args),encoding='utf-8')
command=f'call "{vcvars}" >nul && cl.exe @"{rsp}"'
batch=OUT/'build.cmd'
batch.write_text('@echo off\n'+command+'\n',encoding='utf-8')
with (OUT/'build.log').open('w',encoding='utf-8') as log:
    result=subprocess.run(['cmd.exe','/d','/c',str(batch)],cwd=OUT,stdout=log,stderr=subprocess.STDOUT)
print(json.dumps({'returncode':result.returncode,'log':str(OUT/'build.log'),'dll':str(OUT/'geometry_p2.dll')}))
if not result.returncode: assert (OUT/'geometry_p2.dll').is_file(), 'Expected bridge binary missing'
if result.returncode: print((OUT/'build.log').read_text(encoding='utf-8',errors='replace')[-6000:])
raise SystemExit(result.returncode)
