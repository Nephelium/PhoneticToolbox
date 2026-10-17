"""Real bounded Linux EGG transient sessions and standard parameter-table reads."""
from pathlib import Path
import hashlib,io,json,sys
import sqlite3
import numpy as np
import soundfile as sf
from openpyxl import Workbook
from ptb_api.egg_models import EggTaskConfig
from ptb_worker.egg_interactive import InteractivePreview
from ptb_worker.parameter_preview import render
from ptb_worker.native.linux_runtime import load_profile
from ptb_worker.spectrogram_preview import PreviewError

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    profile,_=load_profile();fs=16000;t=np.arange(fs)/fs
    audio=.15*np.sin(2*np.pi*150*t)+.04*np.sin(2*np.pi*300*t)
    egg=.35*np.sin(2*np.pi*150*t)+.08*np.sin(2*np.pi*450*t)
    wav=io.BytesIO();sf.write(wav,np.column_stack([egg,audio]),fs,format='WAV',subtype='PCM_16');raw=wav.getvalue()
    service=InteractivePreview();checks=[]
    try:
        opened=service.open('synthetic-owner',raw);token=opened['session_id'];assert opened['sha256']==hashlib.sha256(raw).hexdigest();checks.append('stereo_open')
        previous_channel=None
        for label,fields in [('default',{}),('swap',{'flip_channels':True}),('roi',{'roi_start':.2,'roi_end':.6,'micro_center':.4}),('praat_reaper',{'f0_policy':'audio-f0/2','keep_praat_f0':True,'keep_reaper_f0':True})]:
            cfg=EggTaskConfig(mode='preview',**fields).model_dump();value=service.update('synthetic-owner',token,cfg)
            assert value['input_sha256']==opened['sha256'] and value['sample_count']==fs and value['sample_rate_hz']==fs
            channel=cfg['flip_channels']
            assert value['preview'] and value['psd_base64']
            assert bool(value['audio_base64'])==(previous_channel is None or previous_channel!=channel)
            previous_channel=channel;checks.append(label)
        try:service.update('other-owner',token,cfg);raise AssertionError('cross-owner session')
        except PreviewError as e:assert str(e)=='egg_preview_expired'
        service.release('synthetic-owner',token);assert service.linux_raw is None;checks.append('owner_release')
    finally:service.close()
    wb=Workbook();sheet=wb.active;sheet.append(['Time_s','F0_Hz']);sheet.append([0.,150.]);sheet.append([.01,None]);sheet.append([.02,155.]);buffer=io.BytesIO();wb.save(buffer);wb.close()
    table=render(buffer.getvalue(),'result.xlsx');assert table['sha256']==hashlib.sha256(buffer.getvalue()).hexdigest();assert 'F0_Hz' in table['columns'];checks.append('xlsx_parameter_preview')
    with sqlite3.connect(':memory:') as connection:
        connection.execute('CREATE TABLE params(Time_s REAL,pF0 REAL,TextGrid TEXT)')
        connection.executemany('INSERT INTO params VALUES(?,?,?)',[(0.,100.,'阴平 ə'),(.01,None,'上声 ɕ')]);connection.commit();raw=connection.serialize()
    table=render(raw,'result.ptb.sqlite');assert table['rows']==[[0.,100.,'阴平 ə'],[.01,None,'上声 ɕ']] and table['kinds']==['number','number','text'];checks.append('sqlite_parameter_preview_null_and_ipa')
    report=dict(success=True,profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),checks=checks)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
