"""Desktop capability and annotation-lock review regressions."""
import os
import subprocess
import sys
import pytest
from ptb_desktop.annotation import _save_lock, _process_alive
from ptb_desktop.directory_io import pin_directory
from ptb_desktop.file_provider import FileAccessError


def test_annotation_lock_keeps_live_or_unknown_owner(tmp_path):
    path=tmp_path/'.ptb-annotation-test.lock'
    with pin_directory(tmp_path) as output:
        stream=_save_lock(output,path)
        try:
            assert path.read_text()==str(os.getpid())
            with pytest.raises(FileAccessError):_save_lock(output,path)
        finally:stream.close()
    path.write_bytes(b'')
    with pin_directory(tmp_path) as output:
        with pytest.raises(FileAccessError):_save_lock(output,path)


def test_annotation_recovers_provably_exited_owner(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','pass'],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    child.wait(timeout=10)
    assert not _process_alive(child.pid)
    path=tmp_path/'.ptb-annotation-test.lock';path.write_text(str(child.pid))
    with pin_directory(tmp_path) as output:
        stream=_save_lock(output,path)
        try:assert path.read_text()==str(os.getpid())
        finally:stream.close()


def test_file_read_and_list_use_existing_capability_provider(tmp_path):
    import base64
    from ptb_desktop.file_provider import FileProvider
    from ptb_desktop.task_bridge import TaskBridge
    (tmp_path/'input.wav').write_bytes(b'owned bytes')
    provider=FileProvider();grant=provider.choose('input',lambda:tmp_path)
    bridge=TaskBridge.__new__(TaskBridge);bridge.provider=provider
    files=bridge.invoke({'op':'file_list','id':grant['id']})
    value=bridge.invoke({'op':'file_read','id':files[0]['id']})
    assert base64.b64decode(value['base64'])==b'owned bytes'
    with pytest.raises(FileAccessError):bridge.invoke({'op':'file_read','id':'foreign'})


@pytest.mark.skipif(sys.platform!='win32',reason='Windows process ancestry verification')
def test_local_api_restarts_its_crashed_worker_and_runs_next_job(tmp_path):
    import json,time,signal
    from pathlib import Path
    from uuid import uuid4
    from ptb_desktop.local_service import LocalService
    from ptb_worker.local_workspace import prepare_workspace
    from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT
    from ptb_api.job_models import JobInput
    root=Path(__file__).resolve().parents[2]
    db,files=prepare_workspace(tmp_path/'workspace',root/'backend/migrations')
    service=LocalService(db,local_files_root=files)
    def workers():
        # Windows venv Python can be a redirector with another owned interpreter
        # below it. Restrict traversal to this exact API process tree.
        script=f"$all=@(Get-CimInstance Win32_Process); $ids=[Collections.Generic.HashSet[int]]::new(); [void]$ids.Add({service.process.pid}); do {{$added=$false; foreach($p in $all) {{if($ids.Contains([int]$p.ParentProcessId) -and $ids.Add([int]$p.ProcessId)) {{$added=$true}}}}}} while($added); $all | Where-Object {{$ids.Contains([int]$_.ProcessId)}} | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
        result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW,check=True)
        rows=json.loads(result.stdout) if result.stdout.strip() else []
        if isinstance(rows,dict):rows=[rows]
        matches=[row for row in rows if '-m ptb_worker.cli' in (row['CommandLine'] or '')]
        parents={row['ParentProcessId'] for row in matches}
        return [row['ProcessId'] for row in matches if row['ProcessId'] not in parents]
    try:
        service.start();first=workers();assert len(first)==1
        assert first[0] in workers()
        os.kill(first[0],signal.SIGTERM)
        deadline=time.monotonic()+25;restarted=[]
        while time.monotonic()<deadline:
            restarted=[pid for pid in workers() if pid!=first[0]]
            if restarted:break
            time.sleep(.2)
        assert len(restarted)==1
        store=SQLiteJobStore(db)
        job=store.submit('local',JobInput(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex))
        deadline=time.monotonic()+25
        while time.monotonic()<deadline:
            result=store.get('local',job['id'])
            if result['state']=='succeeded':break
            assert result['state'] in ('queued','running'),result
            time.sleep(.1)
        assert result['state']=='succeeded'
        assert service.get('/api/v1/health')['mode']=='local'
    finally:service.close()
