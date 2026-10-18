"""Build the opt-in source extension as an isolated native reference library.

Uses the existing MSVC toolchain. Does not install or replace desktop resources.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build(output):
    output=output.resolve();source=output/'source';source.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'resources/vocal_tract/sources/VTL2.4-API-source.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!='aa696bf4cf5c44cbd433304943aebb241e5ea3064b949daf5ea09f96567cff9b':raise ValueError('VTL source identity mismatch')
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            p=Path(name)
            if p.parent.as_posix()=='Developer/Sources' and p.suffix in ('.cpp','.h'):(source/p.name).write_bytes(bundle.read(name))
    runpy.run_path(str(ROOT/'resources/vocal_tract/sources/m10_source_perturbation.py'))['apply'](source)
    vswhere=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    vs=subprocess.check_output([str(vswhere),'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip()
    vcvars=Path(vs)/'VC/Auxiliary/Build/vcvars64.bat'
    if not vcvars.is_file():raise RuntimeError('Existing MSVC compiler unavailable')
    # The existing upstream source uses SoundLib in its public test entry point.
    args=['/nologo','/LD','/O2','/EHsc','/std:c++14','/DWIN32','/D_USRDLL','/D_USE_MATH_DEFINES','/MD',f'/I"{source}"']
    args += [f'"{p}"' for p in sorted(source.glob('*.cpp'))]+['/FeVocalTractLabApi.dll','/link','winmm.lib']
    rsp=output/'build.rsp';rsp.write_text('\n'.join(args),encoding='utf-8')
    batch=output/'build.cmd';batch.write_text(f'@echo off\ncall "{vcvars}" >nul\ncl.exe @"{rsp}"\n',encoding='utf-8')
    with (output/'build.log').open('w',encoding='utf-8') as log:
        subprocess.run(['cmd.exe','/d','/c',str(batch)],cwd=output,stdout=log,stderr=subprocess.STDOUT,check=True)
    native=output/'native';native.mkdir(exist_ok=True)
    for target in ('VocalTractLabApi.dll','VocalTractLabAnalysis.dll'):shutil.copyfile(output/'VocalTractLabApi.dll',native/target)
    for name in ('geometry_p2.dll','JD2.speaker','M01.speaker','W02.speaker'):shutil.copyfile(ROOT/'resources/vocal_tract/native'/name,native/name)
    report={'platform':'windows-x64','scope':'isolated source reference; desktop runtime not installed',
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in native.iterdir() if p.is_file()}}
    (output/'build-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    build(parser.parse_args().output_dir)
