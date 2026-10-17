"""Use existing WSL g++ only; no Python/dependency installation inside WSL."""
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from workbench_source import bind_sources
from verification_artifacts import verification_run,write_json
bind_sources(ROOT)
from phonetic_core.vocal_tract.engine import Engine,dp


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--distribution',default='NInfer');parser.add_argument('--resources',type=Path,default=ROOT/'output/validation/web-m10/native')
    args=parser.parse_args();resources=args.resources.resolve()
    def linux(p):
        p=Path(p).resolve();return '/mnt/'+p.drive[0].lower()+'/'+p.as_posix()[3:]
    with verification_run('web-m10','linux-abi','WSL C++ direct VTL ABI: validated a/i/u, 33/65 sections, area/length geometry and finite nonzero 9600 samples; no Linux HTTP claim') as (out,scratch):
        engine=Engine(resource_dir=resources);engine.enable_web_sections();cases=[];lines=[]
        try:
            lines.append(f'6 {engine.ntract} {engine.nglottis} {engine.ntube}')
            for count in (33,65):
                engine.set_section_count(count)
                for name in ('a','i','u'):
                    p=engine.validated(engine.presets[name]);g=engine.glottis(f0=125);state=engine.snapshot(p);tube=engine.prepare_tube(p)
                    center=np.empty((count,5));geometric=np.empty(count);acoustic=np.empty(engine.ntube)
                    assert engine.geom.p0_sections(dp(center),dp(geometric),dp(acoustic))==count
                    engine.reset_tube(tube,g);audio=np.concatenate([engine.block_tube(tube,g,960) for _ in range(10)])
                    lines.extend([f'{name} {count}',' '.join(map(repr,p.tolist())),' '.join(map(repr,g.tolist()))])
                    cases.append(dict(name=name,section_count=count,areas=tube['areas'].tolist(),lengths=tube['lengths'].tolist(),geometric=geometric.tolist(),rms=float(np.sqrt(np.mean(audio**2))),peak=float(np.max(np.abs(audio)))))
        finally:engine.close()
        source=scratch/'input.txt';source.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');binary=scratch/'probe';target=out/'linux.json'
        compile=['g++','-std=c++14','-O2',linux(ROOT/'tests/support/vocal_abi_probe.cpp'),'-ldl','-o',linux(binary)]
        with (out/'compile.log').open('wb') as log:subprocess.run(['wsl.exe','-d',args.distribution,'--exec',*compile],stdout=log,stderr=subprocess.STDOUT,check=True)
        shell='ulimit -v 1048576; ulimit -t 30; exec '+shlex.join([linux(binary),linux(resources),linux(source),linux(target)])
        with (out/'probe.log').open('wb') as log:subprocess.run(['wsl.exe','-d',args.distribution,'--exec','sh','-c',shell],stdout=log,stderr=subprocess.STDOUT,check=True)
        actual=json.loads(target.read_text('utf-8'))['cases'];comparisons=[]
        for expected,value in zip(cases,actual,strict=True):
            assert expected['name']==value['name'] and expected['section_count']==value['section_count']
            errors={key:float(np.max(np.abs(np.array(expected[key])-value[key]))) for key in ('areas','lengths','geometric')}
            for key,error in errors.items():np.testing.assert_allclose(value[key],expected[key],rtol=1e-7,atol=1e-10,err_msg=key)
            comparisons.append(dict(name=value['name'],section_count=value['section_count'],geometry_max_abs=errors,linux_rms=value['rms'],windows_rms=expected['rms'],rms_relative_difference=abs(value['rms']/expected['rms']-1)))
        write_json(out/'report.json',dict(success=True,scope='Direct Linux VTL C ABI in WSL, not Linux Python/HTTP/capacity',comparisons=comparisons));print(str(out))


if __name__=='__main__':main()
