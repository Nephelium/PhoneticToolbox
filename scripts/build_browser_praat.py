"""Build original Praat 6.1.38 using task-local Emscripten, CMake and Ninja."""
from pathlib import Path
import argparse,hashlib,os,subprocess,json

ROOT=Path(__file__).resolve().parents[1]


def build(work):
    source=work/'praat_parselmouth-0.4.7'
    sdist=work/'praat_parselmouth-0.4.7.tar.gz'
    if hashlib.sha256(sdist.read_bytes()).hexdigest()!='6dd81d246ce1eef5fd93d8cbdaf1bef61ca40ef1d2fc12aa23996a28071181e6':raise ValueError('Parselmouth source identity')
    sdk=work/'tools/emsdk-4.0.9';cmake=work/'tools/cmake/data/bin/cmake.exe'
    ninja=work/'tools/ninja-1.11.1.4.data/scripts/ninja.exe';target=work/'praat-build'
    env={**os.environ,'EM_CONFIG':str(sdk/'.emscripten'),'EMSDK':str(sdk),'EMSDK_NODE':str(sdk/'node/20.18.0_64bit/bin/node.exe')}
    commands=[[str(cmake),'-G','Ninja','-S',str(ROOT/'packages/phonetic_core/browser'),'-B',str(target),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_MAKE_PROGRAM='+str(ninja),'-DPARSELMOUTH_SOURCE='+source.as_posix(),'-DCMAKE_TOOLCHAIN_FILE='+(sdk/'upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake').as_posix(),'-DFMT_TEST=OFF','-DFMT_DOC=OFF'],[str(cmake),'--build',str(target),'--target','ptb_praat','--parallel','8']]
    with (work/'praat-build.log').open('w',encoding='utf-8') as log:
        for command in commands:
            print('running',command[1:3],flush=True)
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [target/'ptb_praat.js',target/'ptb_praat.wasm']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,required=True);build(parser.parse_args().work.resolve())
