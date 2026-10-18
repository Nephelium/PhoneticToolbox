"""Verify real portable bytes with owned profile, startup download and local files."""
import argparse,hashlib,json,os,shutil,socket,subprocess,time
from pathlib import Path
from uuid import uuid4
from verification_artifacts import verification_run, no_links, inventory
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'release'))
from read_distribution import DistributionReader
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True);parser.add_argument('--delete',action='store_true');parser.add_argument('--relocate',action='store_true');parser.add_argument('--reader-updates',action='store_true');parser.add_argument('--pdf',choices=('probe','verify'));parser.add_argument('--pdf-source',type=Path);args=parser.parse_args()
    if args.pdf and not args.pdf_source:parser.error('--pdf requires a caller-owned --pdf-source')
    package=args.package.resolve();assert package.is_relative_to(ROOT/'dist')
    content=ROOT/'output/paper-reading/2408.05641v1';catalog=json.loads((content/'publish/catalog.json').read_text('utf8'))
    if args.reader_updates: catalog=json.loads((ROOT/'output/paper-reading/intro/stage/catalog.json').read_text('utf8'))
    with verification_run('m18','frozen-r4' if args.delete else 'frozen-r1-receive',recipe={'package':str(package),'delete':args.delete,'network':'Live HTTPS; startup automatic check'}) as (out,scratch):
        # Native DLL dependencies cannot all load from deeply nested QA paths.
        # This unique short directory belongs to this run and is removed last.
        external_root=Path('D:/PTB-Compact-QA-20261006').resolve()
        external=external_root/('m18-'+uuid4().hex[:8]);no_links(external);external.mkdir()
        (external/'owner.json').write_text(json.dumps({'run':str(out),'pid':os.getpid()}),'utf8')
        profile=external/'p';state=profile/'PhoneticToolbox/v3/papers';files=state/'files';files.mkdir(parents=True)
        (state/'first-launch.json').write_text('{"date":"2026-10-07"}','utf8')
        (state/'catalog.json').write_bytes((content/'current-catalog.json').read_bytes())
        previous=json.loads((content/'current-catalog.json').read_text('utf8'))
        for p in previous['papers']:
            for k in ('original','translation','analysisPdf'):
                a=p[k];shutil.copyfile(ROOT/'output/paper-reading/m18-r3/publish'/a['path'],files/(a['sha256']+'.pdf'))
        library=scratch/'library';folder=library/'构音研究';folder.mkdir(parents=True)
        shutil.copyfile(content/'original.pdf',folder/'本地论文.pdf');(folder/'说明.txt').write_text('Preserve all files','utf8');(folder/'补充材料').mkdir();(folder/'补充材料/info.txt').write_text('Preserve nested folders','utf8');(library/'说明.txt').write_text('Collision fixture','utf8')
        if args.pdf:
            shutil.copyfile(args.pdf_source.resolve(),library/'ExtIPA_chart_(2025).pdf')
        (state/'preferences.json').write_text(json.dumps({'autoReceive':not args.reader_updates,'libraryRoot':str(library),'checkIntervalHours':6,'recommendationFolders':['协同发音'],'recommendationAssignments':{'arxiv-2408-05641-v1':'协同发音'}},ensure_ascii=False),'utf8')
        temp=external/'t';temp.mkdir();updates=profile/'PhoneticToolbox/v3/updates';updates.mkdir(exist_ok=True);(updates/'state.json').write_text('{"autoCheck":false}','utf8')
        executable=package/json.loads((package/'build-info.json').read_text('utf8'))['exe']
        if args.relocate:
            payload=external/'payload';payload.mkdir();shutil.copyfile(executable,payload/executable.name)
            assert digest(executable)==digest(payload/executable.name);executable=payload/executable.name
        distribution=DistributionReader(executable)
        from portable_cache_qa import seed_portable_profile
        cache=seed_portable_profile(profile)/'startup-cache'
        native_paths=[str(cache/family/value['id'][:32]/name) for family,value in distribution.cache['components'].items() for name in value['files'] if name.endswith(('.pyd','.dll'))]
        assert max(map(len,native_paths))<260, 'QA profile is too deep for native DLL loading'
        # The environment owns both application data and restoration cache.
        env={k:v for k,v in os.environ.items() if not k.startswith(('PTB_','PYTHON','CONDA','_PYI')) and k not in ('VIRTUAL_ENV','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','QTWEBENGINEPROCESS_PATH','QTWEBENGINE_RESOURCES_PATH','QTWEBENGINE_LOCALES_PATH')}
        with socket.socket() as listener:listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
        env.update(LOCALAPPDATA=str(profile),TEMP=str(temp),TMP=str(temp),PATH=os.pathsep.join((os.environ['SystemRoot']+'/System32',os.environ['SystemRoot'])),QT_QPA_PLATFORM='offscreen',QTWEBENGINE_REMOTE_DEBUGGING=str(port),QTWEBENGINE_CHROMIUM_FLAGS='--mute-audio --disable-gpu --autoplay-policy=no-user-gesture-required',PTB_OWNED_BOOTSTRAP_LOG=str(out/'bootstrap-error.log'))
        plan={'port':port,'verifyDelete':args.delete,'pdfMode':args.pdf,'library':str(library),'report':str(out/'ui-report.json')};(out/'plan.json').write_text(json.dumps(plan),'utf8')
        report={'success':False,'maxNativePath':max(map(len,native_paths)),'external':str(external),'relocated':args.relocate,'exe':{'path':str(executable),'sha256':digest(executable),'bytes':executable.stat().st_size},'scope':'Actual frozen Windows EXE, isolated offscreen Qt and live HTTPS; no user data touched'}
        with (out/'app.log').open('wb') as log:
            if args.relocate:
                app=subprocess.Popen([str(executable)],cwd=executable.parent,env=env,close_fds=True,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                app=subprocess.Popen([str(executable)],cwd=scratch,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                runner='verify_m18_reader_frozen.mjs' if args.reader_updates else 'verify_m18_frozen.mjs'
                subprocess.run(['C:/Program Files/nodejs/node.exe',str(ROOT/'scripts'/runner),str(out/'plan.json')],check=True,timeout=240)
                app.wait(timeout=30);assert app.returncode==0
                if args.delete:
                    # An actual startup error in an owned run must fail without
                    # a PyInstaller dialog. This fixture is a file, not a folder.
                    blocker=scratch/'not-a-project-folder';blocker.write_text('Owned failure fixture','utf8')
                    failure_log=out/'intentional-startup-error.log'
                    failed_env=dict(env,PTB_OWNED_BOOTSTRAP_LOG=str(failure_log))
                    failed=subprocess.run([str(executable),'--local-root',str(blocker)],cwd=scratch,env=failed_env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=90)
                    assert failed.returncode==1 and failure_log.exists()
                    diagnostic=failure_log.read_text('utf8')
                    assert 'RuntimeError: 本地任务目录尚未初始化完成或不属于本应用；现有文件保持不变。' in diagnostic
                    assert 'guarded_main' in diagnostic and blocker.read_text('utf8')=='Owned failure fixture'
                    report['ownedStartupFailureLogged']=True
                local_catalog=json.loads((state/'catalog.json').read_text('utf8'));assert local_catalog==catalog
                report['assets']=[]
                for p in catalog['papers']:
                    for k in ('original','translation','analysisPdf'):
                        a=p[k];target=files/(a['sha256']+'.pdf');assert target.stat().st_size==a['size'] and digest(target)==a['sha256'];report['assets'].append({'paper':p['id'],'kind':k,'sha256':a['sha256']})
                if args.reader_updates:
                    a=catalog['notice']['pdf'];target=files/(a['sha256']+'.pdf')
                    assert target.stat().st_size==a['size'] and digest(target)==a['sha256']
                    report['assets'].append({'paper':catalog['notice']['id'],'kind':'notice','sha256':a['sha256']})
                if args.delete:
                    assert not folder.exists() and (library/'说明.txt').read_text('utf8')=='Preserve all files'
                    assert (library/'补充材料/info.txt').read_text('utf8')=='Preserve nested folders'
                    assert digest(library/'本地论文.pdf')==digest(content/'original.pdf')
                    prefs=json.loads((state/'preferences.json').read_text('utf8'));assert prefs['recommendationFolders']==[] and 'arxiv-2408-05641-v1' not in prefs['recommendationAssignments']
                assert not list(temp.glob('_MEI*'));report.update(success=True,normalExit=app.returncode,papers=len(local_catalog['papers']),deleteVerified=args.delete)
            finally:
                if app.poll() is None:
                    subprocess.run(['taskkill.exe','/PID',str(app.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW);app.wait(timeout=30)
                (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
                assert external.parent==external_root and external.name.startswith('m18-')
                manifest=inventory(external)
                (out/'external-manifest.json').write_text(json.dumps(manifest),'utf8')
                command="$taskTarget = '"+str(external).replace("'","''")+"'; $ErrorActionPreference='Stop'; if (@(Get-CimInstance Win32_Process | Where-Object { ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($taskTarget)) -or ($_.CommandLine -and $_.CommandLine.Contains($taskTarget) -and $_.ProcessId -ne $PID) }).Count) { throw 'Owned QA processes remain' }; Remove-Item -LiteralPath $taskTarget -Recurse -Force"
                cleanup=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
                (out/'external-cleanup.json').write_text(json.dumps({'status':'deleted' if cleanup.returncode==0 else 'retained','path':str(external),'error':cleanup.stderr}),'utf8')
                assert cleanup.returncode==0, cleanup.stderr
        print(out,flush=True)

if __name__=='__main__':main()
