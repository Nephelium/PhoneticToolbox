"""Isolated Linux candidate build. Never installs or replaces Windows resources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,required=True)
options=parser.parse_args()
if sys.platform!='linux':raise SystemExit('Run on Linux with the existing g++ compiler.')
out=options.output_dir.resolve()
out.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'scripts/build_m10_geometry.py'),'--output-dir',str(out),'--no-install','--prepare-only'],check=True)
archive=ROOT/'resources/vocal_tract/sources/VTL2.4-API-source.zip'
original=out/'original';original.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        p=Path(name)
        if p.parent.as_posix()=='Developer/Sources' and p.suffix in ('.cpp','.h'):
            (original/p.name).write_bytes(z.read(name))
native=out/'native';binary=native/'linux-x86_64';binary.mkdir(parents=True,exist_ok=True)
commands=[]
def compile(name,sources,include,extra=()):
    # C++14 requires an out-of-class definition for this odr-used constexpr array.
    assert 'static constexpr int SINUS_COUPLING_SECTION[NUM_SINUS_SECTIONS]' in (include/'Tube.h').read_text()
    constants=out/(name+'-constants.cpp')
    constants.write_text('#include \"Tube.h\"\nconstexpr int Tube::SINUS_COUPLING_SECTION[Tube::NUM_SINUS_SECTIONS];\n')
    sources=[*sources,constants]
    cmd=['g++','-shared','-fPIC','-O2','-std=c++14','-Wl,-Bsymbolic','-Wl,-z,defs','-I'+str(include),*extra,*map(str,sources),'-o',str(binary/name)]
    commands.append(cmd)
    with (out/(name+'.log')).open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
# SoundLib is the host audio-device driver, unused by the array-based VTL API.
# --no-undefined ensures omitting it cannot leave an unresolved runtime call.
compile('libVocalTractLabApi.so',[p for p in sorted(original.glob('*.cpp')) if p.name!='SoundLib.cpp'],original)
# Distinct inodes and -Bsymbolic keep the two native global states independent.
shutil.copyfile(binary/'libVocalTractLabApi.so',binary/'libVocalTractLabAnalysis.so')
names='Geometry Surface VocalTract Tube Splines XmlNode XmlHelper Dsp Signal IirFilter Constants TlModel Matrix2x2'.split()
compile('libgeometry_p2.so',[ROOT/'resources/vocal_tract/sources/geometry_bridge.cpp',*[out/'source'/(n+'.cpp') for n in names]],out/'source',['-D__declspec(x)=__attribute__((visibility("default")))'])
for name in ('JD2.speaker','M01.speaker','W02.speaker'):shutil.copyfile(ROOT/'resources/vocal_tract/native'/name,native/name)
report={'platform':sys.platform,'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'commands':commands,'files':{str(p.relative_to(native)):hashlib.sha256(p.read_bytes()).hexdigest() for p in native.rglob('*') if p.is_file()},'scope':'native candidate only; no HTTP integration or parity claim'}
(out/'build-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
