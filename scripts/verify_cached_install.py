"""Real OwnedQA installation, cache prewarm, busy guard and uninstallation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop/src'))
from ptb_desktop.startup_cache import lease

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--setup',type=Path,required=True)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();setup=args.setup.resolve()
    if not out.is_relative_to(Path('D:/PTB-Compact-QA-20261006')) or not setup.is_relative_to(ROOT/'output/validation'):
        raise ValueError('Only a dedicated OwnedQA setup and new external QA directory are allowed')
    out.mkdir(parents=True,exist_ok=False)
    profile=out/'profile';temp=out/'temp';installed=out/'installed'
    profile.mkdir();temp.mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('PTB_','PYTHON','CONDA','_PYI')) and k not in ('VIRTUAL_ENV','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','QTWEBENGINEPROCESS_PATH','QTWEBENGINE_RESOURCES_PATH','QTWEBENGINE_LOCALES_PATH')}
    env.update(LOCALAPPDATA=str(profile),TEMP=str(temp),TMP=str(temp),PATH=os.pathsep.join((os.environ['SystemRoot']+'/System32',os.environ['SystemRoot'])),PTB_OWNED_BOOTSTRAP_LOG=str(out/'bootstrap-error.log'))
    def run(command,label):
        start=time.monotonic()
        with (out/(label+'.stdout.log')).open('wb') as log:
            code=subprocess.run([str(x) for x in command],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=600).returncode
        return {'exitCode':code,'seconds':time.monotonic()-start}
    cache=profile/'PhoneticToolbox/v3/startup-cache'
    report={'success':False,'scope':'Real OwnedQA installer/uninstaller, no product registration or shortcuts; Windows-only PATH and private profile'}
    try:
        report['install']=run([setup,'/CURRENTUSER','/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/SP-','/DIR='+str(installed),'/LOG='+str(out/'install.log')],'install')
        assert report['install']['exitCode']==0
        app=installed/'PhoneticToolbox.exe'
        def digest(file):
            with file.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
        report['exeSha256']=digest(app);assert report['exeSha256']==digest(args.exe)
        report['installedFiles']=sorted(p.name for p in installed.iterdir())
        assert set(report['installedFiles'])=={'PhoneticToolbox.exe','application.json','.ptb-installed.json','unins000.exe','unins000.dat'}
        report['installationPrepared']=json.loads((cache/'last-launch.json').read_text('utf8'))
        assert report['installationPrepared']['prepared']==['science','host','apps']
        report['prepareAgain']=run([app,'--ptb-prepare-cache'],'warm')
        assert report['prepareAgain']['exitCode']==0
        report['warm']=json.loads((cache/'last-launch.json').read_text('utf8'));assert report['warm']['prepared']==[] and report['warm']['reused']==['science','host','apps']
        markers=[profile/'PhoneticToolbox/v3/owned-research-project.wav',profile/'PhoneticToolbox-v3/workbench/owned-settings.json']
        for file in markers:file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'RESEARCH AND SETTINGS MUST SURVIVE')
        original={str(f):digest(f) for f in markers}
        uninstaller=installed/'unins000.exe'
        # A real OS file lease, also used by every actual frozen process.
        with lease(cache):
            report['busyUninstall']=run([uninstaller,'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/LOG='+str(out/'busy-uninstall.log')],'busy-uninstall')
            assert app.is_file() and cache.is_dir()
        report['uninstall']=run([uninstaller,'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/LOG='+str(out/'uninstall.log')],'uninstall')
        assert report['uninstall']['exitCode']==0
        assert not app.exists() and not cache.exists()
        assert {str(f):digest(f) for f in markers}==original
        report['researchAndSettingsPreserved']=original
        deadline=time.monotonic()+10
        while installed.exists() and time.monotonic()<deadline:time.sleep(.1)
        report['remainingInstallFiles']=sorted(p.name for p in installed.iterdir()) if installed.exists() else []
        assert not report['remainingInstallFiles']
        report['leftoverBootDirectories']=[p.name for p in temp.glob('_MEI*')];assert not report['leftoverBootDirectories']
        report['success']=True
    finally:(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
