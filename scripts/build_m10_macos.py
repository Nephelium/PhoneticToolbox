"""Build the shared VTL patches on an existing Mac compiler, in a candidate directory."""
import argparse
import hashlib
import json
import platform
import runpy
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA='aa696bf4cf5c44cbd433304943aebb241e5ea3064b949daf5ea09f96567cff9b'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build(output):
    arch=platform.machine().lower()
    if sys.platform!='darwin' or arch not in ('arm64','x86_64'):
        raise RuntimeError('Run on the target Mac with its existing Xcode command line compiler')
    archive=ROOT/'resources/vocal_tract/sources/VTL2.4-API-source.zip'
    if digest(archive)!=SOURCE_SHA:raise ValueError('VTL source identity mismatch')
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_m10_geometry.py'),'--output-dir',str(output),'--no-install','--prepare-only'],check=True)
    original=output/'original';original.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            p=Path(name)
            if p.parent.as_posix()=='Developer/Sources' and p.suffix in ('.cpp','.h'):
                (original/p.name).write_bytes(source.read(name))
    runpy.run_path(str(ROOT/'resources/vocal_tract/sources/m10_source_perturbation.py'))['apply'](original)
    native=output/'native';binary=native/('darwin-'+arch);binary.mkdir(parents=True,exist_ok=True)
    commands=[]
    def compile(name,sources,include,extra=()):
        constants=output/(name+'-constants.cpp')
        constants.write_text('#include "Tube.h"\nconstexpr int Tube::SINUS_COUPLING_SECTION[Tube::NUM_SINUS_SECTIONS];\n',encoding='utf-8')
        command=['/usr/bin/clang++','-dynamiclib','-arch',arch,'-mmacosx-version-min=14.0','-O2','-std=c++14',
                 '-ffp-contract=off','-Wl,-undefined,error','-Wl,-install_name,@rpath/'+name,
                 '-I'+str(include),*extra,*map(str,[*sources,constants]),'-o',str(binary/name)]
        commands.append(command)
        with (output/(name+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['/usr/bin/codesign','--force','--sign','-',str(binary/name)],check=True)
    for name in ('libVocalTractLabApi.dylib','libVocalTractLabAnalysis.dylib'):
        compile(name,[p for p in sorted(original.glob('*.cpp')) if p.name!='SoundLib.cpp'],original)
    names='Geometry Surface VocalTract Tube Splines XmlNode XmlHelper Dsp Signal IirFilter Constants TlModel Matrix2x2'.split()
    compile('libgeometry_p2.dylib',[ROOT/'resources/vocal_tract/sources/geometry_bridge.cpp',*[output/'source'/(n+'.cpp') for n in names]],output/'source',['-D__declspec(x)=__attribute__((visibility("default")))'])
    for name in ('JD2.speaker','M01.speaker','W02.speaker'):shutil.copyfile(ROOT/'resources/vocal_tract/native'/name,native/name)
    files={p.name:digest(p) for p in binary.iterdir() if p.suffix=='.dylib'}
    pair={'schema':'ptb-vtl-native-pair/1','platform':'darwin','architecture':arch,'same_source_sha256':SOURCE_SHA,'files':files}
    (binary/'native-pair.json').write_text(json.dumps(pair,indent=2)+'\n',encoding='utf-8')
    inputs=[Path(__file__),ROOT/'scripts/build_m10_geometry.py',*(ROOT/'resources/vocal_tract/sources').glob('*.py'),ROOT/'resources/vocal_tract/sources/geometry_bridge.cpp']
    report={**pair,'geometry_method':'m10/r21','minimum_macos':'14.0','commands':commands,
            'inputs':{p.relative_to(ROOT).as_posix():digest(p) for p in inputs},
            'scope':'Native candidate only; no installed application or public release changed'}
    (output/'build-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'platform':'darwin','architecture':arch,'files':files}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    build(parser.parse_args().output_dir)
