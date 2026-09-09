import pytest
from ptb_worker.io.limits import LimitedBuffer, LimitError
from ptb_worker.io.limits import Limits, FormatError, Cancelled
from ptb_worker.io.scratch import Scratch
from ptb_worker.native.windows import OwnedProcess,InputPipe,open_process,wait,close
from ptb_worker.native.reaper import Reaper,collect_pipe
from ptb_worker.io.parameter_exports import export_pair,verify_pair,table_from_frame,_build_pair
from phonetic_core.models.audio import AudioInput
from phonetic_core.models.acoustic import AcousticConfig
from phonetic_core.ports.acoustic import AcousticBackends
from phonetic_core.services.acoustic import analyze_audio
from dataclasses import replace
from pathlib import Path
import io
import sys
import time
import threading
import subprocess
import sqlite3
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BINARY=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe'
BASE=sys._base_executable


def dead(pid):
    handle=open_process(0x100000,False,pid)
    if not handle:return True
    try:return wait(handle,0)==0
    finally:close(handle)


def tone():
    phase=2*np.pi*150*np.arange(16000)/16000
    return AudioInput(sum(np.sin(k*phase)/(k*k) for k in range(1,13))*.4,16000)


def frame():
    return pd.DataFrame({'Time_s':[0.,.005,.01,.015], 'pF0':[120.,np.nan,np.inf,-np.inf],
                         'TextGrid':['=1+1','+cmd','@literal','井井 əʊ'], 'text_音节':['','aː','"quote"','\n换行']})


def test_limit_rejects_before_write_or_seek():
    stream=LimitedBuffer(8)
    stream.write(b'1234')
    with pytest.raises(LimitError): stream.write(b'56789')
    assert stream.getvalue()==b'1234'
    with pytest.raises(LimitError): stream.seek(9)


def test_native_real_output_and_cleanup(tmp_path):
    with Scratch(tmp_path,1_000_000) as scratch:
        native=Reaper(BINARY,scratch)
        track=native(tone(),.005,60,880,hilbert=True,no_highpass=False)
        assert len(track.times)>0 and track.actual_backend=='native_reaper'
        np.testing.assert_allclose(np.diff(track.times),.005,rtol=0,atol=1e-12)
        assert abs(np.nanmedian(track.values)-150)<2
        assert dead(native.last_pid) and scratch.used==0 and not list(scratch.root.iterdir())
    assert not list(tmp_path.iterdir())


def test_native_budgets_binary_and_direct_path_refusal(tmp_path):
    with pytest.raises(TypeError):Reaper(BINARY,tmp_path)
    with Scratch(tmp_path,1000000) as scratch:
        with pytest.raises(ValueError):Reaper(Path(__file__),scratch)
        native=Reaper(BINARY,scratch,replace(Limits(),output_bytes=128))
        with pytest.raises(LimitError):native(tone(),.005,60,880,hilbert=True,no_highpass=False)
        tiny_rate=AudioInput(np.ones(100),1)
        with pytest.raises(LimitError):native(tiny_rate,.005,60,880,hilbert=True,no_highpass=False)
        assert scratch.used==0 and not list(scratch.root.iterdir())
    with Scratch(tmp_path,10) as scratch:
        with pytest.raises(LimitError):Reaper(BINARY,scratch)(tone(),.005,60,880,hilbert=True,no_highpass=False)
        assert scratch.used==0


@pytest.mark.parametrize('mode',['cancel','timeout','exit','abrupt'])
def test_owned_faults_terminate_only_task_process(tmp_path,mode):
    other=subprocess.Popen([BASE,'-c','import time;time.sleep(30)'],creationflags=subprocess.CREATE_NO_WINDOW)
    pipe=InputPipe();pids=[];start=time.monotonic()
    script='import time;time.sleep(30)' if mode in ('cancel','timeout') else 'raise SystemExit(7)' if mode=='exit' else 'import os;os._exit(9)'
    limits=replace(Limits(),timeout_seconds=.3 if mode=='timeout' else 5.)
    try:
        expected=Cancelled if mode=='cancel' else LimitError if mode=='timeout' else FormatError
        with pytest.raises(expected):
            collect_pipe([BASE,'-c',script],pipe,tmp_path,limits,
                stop=lambda:mode=='cancel' and time.monotonic()-start>.15,on_started=pids.append)
        assert pids and dead(pids[0]) and other.poll() is None
    finally:other.terminate();other.wait(timeout=5)


def test_job_memory_limit_is_enforced(tmp_path):
    pipe=InputPipe()
    script="import sys\ntry:\n x=bytearray(512_000_000)\n message=b'unbounded'\nexcept MemoryError:\n message=b'bounded'\nwith open(sys.argv[1],'wb',buffering=0) as f:f.write(message)"
    result,pid=collect_pipe([BASE,'-c',script,pipe.name],pipe,tmp_path,replace(Limits(),process_bytes=64_000_000))
    assert result==b'bounded' and dead(pid)


def test_job_close_terminates_descendant(tmp_path):
    marker=tmp_path/'owned-child-id.txt'
    script="import subprocess,sys,time,pathlib\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\npathlib.Path(sys.argv[1]).write_text(str(p.pid))\ntime.sleep(30)"
    process=OwnedProcess([BASE,'-c',script,str(marker)],tmp_path,128000000)
    child=None
    try:
        until=time.monotonic()+5
        while not marker.exists() and time.monotonic()<until:time.sleep(.01)
        child=int(marker.read_text());assert process.owns_pid(child)
    finally:process.close()
    assert dead(process.pid) and dead(child)


def test_pipe_rejects_unrelated_client(tmp_path):
    owner=OwnedProcess([BASE,'-c','import time;time.sleep(5)'],tmp_path,128000000)
    pipe=InputPipe();errors=[]
    def foreign():
        try:
            with open(pipe.name,'wb',buffering=0) as f:f.write(b'foreign')
        except OSError as exc:errors.append(type(exc).__name__)
    thread=threading.Thread(target=foreign,daemon=True);thread.start()
    try:
        until=time.monotonic()+3
        with pytest.raises(ValueError,match='Unexpected pipe client'):
            while time.monotonic()<until:pipe.read(64,owner);time.sleep(.01)
    finally:pipe.close();owner.close();thread.join(timeout=2)
    assert not thread.is_alive()


@pytest.mark.parametrize('exc',[Cancelled('cancelled'),LimitError('budget')])
def test_core_never_swallows_native_abort(exc):
    def abort(*a,**k):raise exc
    with pytest.raises(type(exc)):
        analyze_audio(tone(),AcousticConfig(selected_parameter_keys=('rF0',)),backends=AcousticBackends(reaper=abort))


def test_real_pair_files_readback_and_literal_formula(tmp_path):
    table=table_from_frame(frame(),Limits())
    with Scratch(tmp_path,1_000_000) as scratch:
        pair=export_pair(frame(),scratch)
        assert scratch.used==0
        xlsx=scratch.create(pair.xlsx,'.xlsx');db=scratch.create(pair.sqlite,'.sqlite')
        verify_pair(type(pair)(xlsx.read_bytes(),db.read_bytes(),pair.columns,pair.row_count),table)
        from openpyxl import load_workbook
        wb=load_workbook(xlsx,data_only=False)
        assert wb.active['C2'].value=='=1+1' and wb.active['C2'].data_type=='s';wb.close()
        conn=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
        try:
            assert conn.execute('SELECT "TextGrid" FROM params ORDER BY rowid').fetchall()==[(v,) for v in frame()['TextGrid']]
            assert conn.execute('PRAGMA integrity_check').fetchone()==('ok',)
        finally:conn.close()
    assert not list(tmp_path.iterdir())


def test_xlsx_never_opens_temporary_disk_file(monkeypatch):
    import openpyxl.worksheet._writer as writer
    monkeypatch.setattr(writer,'create_temporary_file',lambda *a,**k:pytest.fail('Unaccounted temporary file'))
    _build_pair(table_from_frame(frame(),Limits()),Limits())


@pytest.mark.parametrize('mode',['second_output','memory','cancel','scratch','cells','text'])
def test_export_failure_returns_no_partial_pair_and_cleans(tmp_path,mode):
    limits=Limits();data=frame();budget=1_000_000;started=[]
    if mode=='second_output':limits=replace(limits,output_bytes=10000)
    if mode=='memory':limits=replace(limits,process_bytes=16_000_000)
    if mode=='scratch':budget=10
    if mode=='cells':limits=replace(limits,cells=4)
    if mode=='text':data.loc[0,'TextGrid']='x'*32768
    with Scratch(tmp_path,budget) as scratch:
        with pytest.raises((LimitError,FormatError,Cancelled)):
            export_pair(data,scratch,limits,stop=lambda:mode=='cancel' and bool(started),on_started=started.append)
        assert scratch.used==0 and not list(scratch.root.iterdir())
        assert all(dead(pid) for pid in started)


def test_export_rejects_empty_duplicate_and_unsorted():
    for data in [pd.DataFrame({'Time_s':[]}),pd.DataFrame([[0.,1.]],columns=['Time_s','time_s']),pd.DataFrame({'Time_s':[1.,0.]})]:
        with pytest.raises(FormatError):table_from_frame(data,Limits())
