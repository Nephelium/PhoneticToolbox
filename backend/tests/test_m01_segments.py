"""Actual Windows child and output format checks, not public task acceptance."""
from dataclasses import replace
import hashlib
import io
import json
import struct
import time
import pytest
import numpy as np
from scipy.io import wavfile
from ptb_worker.segmentation import prepare_segments,unpack_bundle,SEGMENT_LIMITS
from ptb_worker.io.scratch import Scratch
from ptb_worker.io.limits import FormatError,Cancelled,LimitError
from ptb_worker.native.windows import open_process,wait,close


def audio():
    samples=np.column_stack((np.arange(1000,dtype=np.int16),-np.arange(1000,dtype=np.int16)))
    f=io.BytesIO();wavfile.write(f,1000,samples);return f.getvalue(),samples


def grid(labels=('ɑ̃','sil','b')):
    text='File type = "ooTextFile short"\n"TextGrid"\n0\n1\n<exists>\n1\n"IntervalTier"\n"音节"\n0\n1\n3\n'
    for a,b,label in zip((0,.2,.5),(.2,.5,1),labels):text+=f'{a}\n{b}\n"{label}"\n'
    return text.encode()


def dead(pid):
    handle=open_process(0x100000,False,pid)
    if not handle:return True
    try:return wait(handle,0)==0
    finally:close(handle)


def parent_result(wav):
    from ptb_api.acoustic_models import AcousticResult,AcousticConfigSnapshot,config_digest
    from phonetic_core.catalog import PARAMETER_MAPPING
    config=AcousticConfigSnapshot(selection={'keys':['pF0']},settings={'frameshift_ms':100})
    return AcousticResult(metadata=dict(core_version='3.0.0a1',project_id='00000000-0000-4000-8000-000000000001',
        inputs=[dict(asset_id='00000000-0000-4000-8000-000000000002',sha256=hashlib.sha256(wav).hexdigest(),role='audio',expires_at=None)],
        decoded=dict(sample_rate_hz=1000,channels=2,sample_count=1000,sample_dtype='int16'),config=config,config_sha256=config_digest(config),
        source_ids=['SRC-PRAAT'],backends=[]),times_s=[i*.1 for i in range(10)],
        numeric=[dict(key='pF0',label=PARAMETER_MAPPING['pF0'],unit='Hz',scope='catalog',values=[120.,None,*([130.]*8)],nonfinite=[0,1,*([0]*8)],reason=[None,'legacy_nonfinite_unknown',*([None]*8)])],
        text=[dict(key='text_音节',values=['=1+1']*10)],column_order=['Time_s','pF0','text_音节']).model_dump_json().encode()


def test_actual_stereo_segmentation_roundtrip_and_child_cleanup(tmp_path):
    wav,samples=audio();started=[];beats=[]
    with Scratch(tmp_path,10_000_000) as scratch:
        bundle=prepare_segments(wav,grid(),'音节',scratch,audio_name='井井.wav',on_started=started.append,heartbeat=lambda:beats.append(time.monotonic()))
        assert scratch.used==0
    assert len(started)==1 and dead(started[0]) and beats
    assert len(bundle.payloads)==2
    for raw,expected in zip(bundle.payloads,(samples[:200],samples[500:])):
        fs,values=wavfile.read(io.BytesIO(raw));assert fs==1000;np.testing.assert_array_equal(values,expected)
    assert [s['first_sample'] for s in bundle.manifest['segments']]==[0,500]
    assert all(s['parameter_status']=='no_parent_result' for s in bundle.manifest['segments'])
    assert bundle.manifest['reestimated'] is False


def test_matching_parent_exports_xlsx_sqlite_with_original_and_local_time(tmp_path):
    import sqlite3
    from openpyxl import load_workbook
    wav,_=audio()
    with Scratch(tmp_path,10_000_000) as scratch:
        bundle=prepare_segments(wav,grid(),'音节',scratch,parent_result=parent_result(wav))
    assert [f['format'] for f in bundle.manifest['files']]==['wav','xlsx','sqlite']*2
    conn=sqlite3.connect(':memory:')
    try:
        conn.deserialize(bundle.payloads[5]);rows=conn.execute('SELECT * FROM params').fetchall()
        assert rows[0][:3]==(0.,.5,130.) and rows[-1][1]==.9
    finally:conn.close()
    wb=load_workbook(io.BytesIO(bundle.payloads[1]),read_only=True,data_only=False)
    try:
        rows=list(wb.active);assert rows[1][-1].value=='=1+1' and rows[1][-1].data_type=='s'
        assert rows[2][2].value is None
    finally:wb.close()


@pytest.mark.parametrize('case',['mismatch','missing','silent','budget'])
def test_source_layer_and_budget_errors_do_not_return_partial_bundle(tmp_path,case):
    wav,_=audio();parent=parent_result(wav) if case=='mismatch' else None
    if parent:
        value=json.loads(parent);value['metadata']['inputs'][0]['sha256']='0'*64;parent=json.dumps(value).encode()
    expected={'mismatch':'parent_result_source_mismatch','missing':'missing_or_invalid_tier','silent':'no_labelled_segments','budget':'segment_budget_exceeded'}[case]
    with Scratch(tmp_path,10_000_000) as scratch:
        with pytest.raises(FormatError,match=expected):
            prepare_segments(wav,grid(('sil','eps','')) if case=='silent' else grid(),'wrong' if case=='missing' else '音节',scratch,
                parent_result=parent,limits=replace(SEGMENT_LIMITS,output_bytes=2000) if case=='budget' else SEGMENT_LIMITS)
        assert scratch.used==0


def test_cancel_timeout_and_failed_heartbeat_terminate_owned_child(tmp_path):
    wav,_=audio()
    for case in ('cancel','timeout','lease'):
        started=[];beats=[]
        def heartbeat():
            beats.append(1)
            if started and case=='lease':raise RuntimeError('lease_lost')
        with Scratch(tmp_path,10_000_000) as scratch:
            with pytest.raises((Cancelled,LimitError,RuntimeError)):
                prepare_segments(wav,grid(),'音节',scratch,on_started=started.append,
                    stop=lambda:bool(started) and case=='cancel',heartbeat=heartbeat,
                    limits=replace(SEGMENT_LIMITS,timeout_seconds=.01) if case=='timeout' else SEGMENT_LIMITS)
            assert scratch.used==0
        assert started and dead(started[0])


def test_malformed_bundle_cannot_smuggle_output_path_or_extra_bytes():
    entry={'name':'../a.wav','format':'wav','size_bytes':1,'sha256':hashlib.sha256(b'a').hexdigest()}
    manifest=json.dumps({'kind':'prepared_segments','files':[entry]}).encode()
    with pytest.raises(FormatError):unpack_bundle(struct.pack('<Q',len(manifest))+manifest+b'a',10000)


@pytest.mark.parametrize('value',[None,[],{'kind':'prepared_segments','files':None},
    {'kind':'prepared_segments','files':[None]}, {'kind':'prepared_segments','files':[{}]},
    {'error':[]}, {'kind':'prepared_segments','files':[{'format':{}}]}])
def test_malformed_manifest_fails_as_format_error(value):
    raw=json.dumps(value).encode()
    with pytest.raises(FormatError):unpack_bundle(struct.pack('<Q',len(raw))+raw,10000)


def test_no_frames_keeps_audio_and_rounded_duplicate_names_remain_unique(tmp_path):
    wav,_=audio()
    # Two tiny labelled intervals round to the same 3-decimal filename times.
    tg=b'File type = "ooTextFile short"\n"TextGrid"\n0\n1\n<exists>\n1\n"IntervalTier"\n"tier"\n0\n1\n2\n.011\n.012\n"same"\n.021\n.022\n"same"\n'
    with Scratch(tmp_path,10_000_000) as scratch:
        bundle=prepare_segments(wav,tg,'tier',scratch,parent_result=parent_result(wav))
    assert len(bundle.payloads)==2 and all(s['parameter_status']=='no_frames' for s in bundle.manifest['segments'])
    fs=100000;values=np.arange(fs,dtype=np.int32);stream=io.BytesIO();wavfile.write(stream,fs,values)
    tg=tg.replace(b'.011\n.012',b'.00011\n.00012').replace(b'.021\n.022',b'.00021\n.00022')
    with Scratch(tmp_path,10_000_000) as scratch:
        bundle=prepare_segments(stream.getvalue(),tg,'tier',scratch)
    assert len({f['name'] for f in bundle.manifest['files']})==2
    assert all('_0.000_0.000_' in f['name'] for f in bundle.manifest['files'])
    assert [wavfile.read(io.BytesIO(raw))[1].tolist() for raw in bundle.payloads]==[[11],[21]]
