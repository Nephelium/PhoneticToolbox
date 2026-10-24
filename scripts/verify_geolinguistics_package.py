"""Run M19 against the actual final EXE and companion data, with isolated state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from portable_cache_qa import seed_portable_profile
from package_geolinguistics_data import DATA_FILES
from verification_artifacts import verification_run, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    if package.parent != ROOT/'dist':
        raise ValueError('Expected an owned final package')
    info = json.loads((package/'build-info.json').read_text('utf-8'))
    source = package/info['exe']
    with source.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest() == info['sha256']
    data_root=package/'geolinguistics-data'
    manifest=json.loads((data_root/'data-manifest.json').read_text('utf-8'))
    assert manifest['profile']=='interactive-data'
    assert {f['path'] for f in manifest['files']}==set(DATA_FILES)
    assert {p.relative_to(data_root).as_posix() for p in data_root.rglob('*') if p.is_file()}==set(DATA_FILES)|{'data-manifest.json'}
    for item in manifest['files']:
        raw=(data_root/item['path']).read_bytes()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    with verification_run('geolinguistics','package','Final EXE, Windows-only PATH, isolated profile and exact companion data') as (out,scratch):
        print(out,flush=True)
        executable = scratch/source.name
        shutil.copyfile(source,executable)
        shutil.copyfile(package/'application.json',scratch/'application.json')
        shutil.copytree(package/'geolinguistics-data',scratch/'geolinguistics-data')
        profile,temp = scratch/'profile',scratch/'temp'
        profile.mkdir();temp.mkdir()
        seed_portable_profile(profile)
        env = {k:v for k,v in os.environ.items() if not k.startswith(('PTB_','PYTHON','CONDA','_PYI','QT_','QTWEBENGINE')) and k!='VIRTUAL_ENV'}
        env.update(LOCALAPPDATA=str(profile),TEMP=str(temp),TMP=str(temp),
                   PATH=os.pathsep.join([os.environ['SystemRoot']+'/System32',os.environ['SystemRoot']]),
                   QT_QPA_PLATFORM='windows',QTWEBENGINE_CHROMIUM_FLAGS='--mute-audio',
                   PTB_OWNED_BOOTSTRAP_LOG=str(out/'bootstrap-error.log'))
        with (out/'process.log').open('wb') as log:
            result = subprocess.run([str(executable),'--verify-m19-preview',str(out)],cwd=scratch,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode == 0, (result.returncode,str(out/'process.log'))
        nested = Path((out/'m19-run-path.txt').read_text('utf-8'))
        report = json.loads((nested/'report.json').read_text('utf-8'))
        assert report['success'] is True
        assert not list(temp.glob('_MEI*')), 'Startup temporary directory retained'
        write_json(out/'report.json',{'success':True,'exeSha256':info['sha256'],'exitCode':result.returncode,
                   'dataManifestSha256':hashlib.sha256((package/'geolinguistics-data/data-manifest.json').read_bytes()).hexdigest(),
                   'scope':'Actual final Windows EXE in owned verification scratch; Windows-only PATH and new profile; default GPU visible Qt; controlled save destination',
                   'detail':str(nested/'report.json')})
        # The frozen verifier does not ship development cleanup scripts. After
        # its process exits, the source-side orchestrator cleans only its run.
        cleanup = subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',str(nested)],capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
        if cleanup.returncode:
            raise RuntimeError('Owned nested cleanup failed: '+cleanup.stdout+cleanup.stderr)


if __name__ == '__main__':
    main()
