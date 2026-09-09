"""Real subprocess protocol and owner-EOF tests; no database claim."""
import json
import subprocess
import sys
from phonetic_core import __version__


def test_core_child_result_and_owner_eof():
    config={'snapshot':{'operation':'pipeline_check','core_version':__version__,'config':{'sample_count':257,'seed':0}}}
    child=subprocess.Popen([sys.executable,'-m','ptb_worker.core_child'],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        child.stdin.write(json.dumps(config)+'\n');child.stdin.flush()
        rows=[json.loads(child.stdout.readline()) for _ in range(4)]
        child.wait(timeout=10)
        assert child.returncode==0 and rows[-1]['result']['sample_count']==257
        assert rows[-1]['result']['complete'] is True
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=5)
        child.stdin.close();child.stdout.close();child.stderr.close()
    child=subprocess.Popen([sys.executable,'-m','ptb_worker.core_child'],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        config['snapshot']['config']['sample_count']=4096;config['step_delay']=0.2
        child.stdin.write(json.dumps(config)+'\n');child.stdin.flush()
        assert 'progress' in json.loads(child.stdout.readline())
        child.stdin.close();child.wait(timeout=5)
        assert child.returncode==75
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=5)
        child.stdout.close()
