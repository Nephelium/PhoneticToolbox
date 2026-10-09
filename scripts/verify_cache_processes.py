"""Interrupt a real launcher, then start two actual copies against one cache."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exe',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();exe=args.exe.resolve()
    if not out.is_relative_to(Path('D:/PTB-Compact-QA-20261006')):raise ValueError('Expected a new owned QA output')
    out.mkdir(parents=True,exist_ok=False);profile=out/'profile';profile.mkdir();temp=out/'temp';temp.mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('PTB_','PYTHON','CONDA','_PYI')) and k!='VIRTUAL_ENV'}
    env.update(LOCALAPPDATA=str(profile),TEMP=str(temp),TMP=str(temp),PATH=os.pathsep.join((os.environ['SystemRoot']+'/System32',os.environ['SystemRoot'])),PTB_OWNED_BOOTSTRAP_LOG=str(out/'bootstrap-error.log'))
    cache=profile/'PhoneticToolbox/v3/startup-cache';children=[]
    def launch(flag='--ptb-prepare-cache'):
        child=subprocess.Popen([str(exe),flag],env=env,cwd=out,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW);children.append(child);return child
    report={'success':False,'scope':'Actual final frozen EXE; intentionally killed owned first preparation; two fresh launcher processes share one private cache'}
    try:
        interrupted=launch();deadline=time.monotonic()+90
        while time.monotonic()<deadline and not any(cache.glob('staging/*/runtimes/egg/python.exe')):
            if interrupted.poll() is not None:raise RuntimeError('First preparation ended before interruption')
            time.sleep(.02)
        assert any(cache.glob('staging/*/runtimes/egg/python.exe'))
        subprocess.run(['taskkill.exe','/PID',str(interrupted.pid),'/T','/F'],check=True,stdout=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        interrupted.wait(timeout=30);assert not list(cache.glob('science/*/ready.json'))
        report['interruptedIncompleteDirectories']=[p.name for p in (cache/'staging').iterdir()]
        report['forcedTerminationTemporaryDirectories']=[p.name for p in temp.glob('_MEI*')]
        first,second=launch(),launch()
        report['simultaneousExitCodes']=[first.wait(timeout=300),second.wait(timeout=300)]
        assert report['simultaneousExitCodes']==[0,0]
        report['afterConcurrent']=json.loads((cache/'last-launch.json').read_text('utf8'))
        assert report['afterConcurrent']['prepared']==[] and report['afterConcurrent']['reused']==['science','host','apps']
        assert not list((cache/'staging').iterdir())
        report['readyDirectories']={family:len(list(cache.glob(family+'/*/ready.json'))) for family in ('science','host','apps')}
        assert set(report['readyDirectories'].values())=={1}
        report['clearExitCode']=launch('--ptb-clear-startup-cache').wait(timeout=180)
        assert report['clearExitCode']==0 and not cache.exists()
        report['success']=True
    finally:
        for child in children:
            if child.poll() is None:subprocess.run(['taskkill.exe','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        (out/'report.json').write_text(json.dumps(report,indent=2),'utf8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
