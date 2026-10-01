"""Real-only EGG cache parity and owned-process checks. Never writes source audio."""
from pathlib import Path
import base64
import hashlib
import json
import struct
import sys
import time
import subprocess
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'backend/src'), str(ROOT/'packages/phonetic_core/src')]
SOURCE = Path(r'C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav')


def main():
    from ptb_worker.egg_child import prepare
    from ptb_worker import egg_preview
    reference_module=types.ModuleType('ptb_worker.egg_preview_reference')
    exec(subprocess.check_output(['git','show','HEAD:backend/src/ptb_worker/egg_preview.py'],cwd=ROOT).decode('utf-8'),reference_module.__dict__)
    updated_preview=egg_preview.preview_files
    from ptb_worker.egg_interactive_child import Session
    raw = SOURCE.read_bytes()
    before = hashlib.sha256(raw).hexdigest()
    session = Session(raw)
    cfg = dict(mode='preview', roi_start=39.747, roi_end=41.096, micro_center=40.4966,
               keep_praat_f0=False, keep_gci_f0=False, glottal_movement=False, lowpass_cutoff=2000)
    changes = [{}, {'micro_center':40.5}, {'micro_width_ms':150}, {'roi_start':40,'roi_end':40.8},
               {'signal_mode':'raw'}, {'signal_mode':'filtered','highpass_cutoff':30},
               {'keep_gci_f0':True}, {'keep_praat_f0':True}, {'keep_praat_f0':False,'keep_gci_f0':False},
               {'flip_channels':True}, {'flip_channels':False,'roi_start':0,'roi_end':.5,'micro_center':0},
               {'roi_start':76.8,'roi_end':len(session.samples)/session.fs,'micro_center':len(session.samples)/session.fs}]
    records=[]
    for change in changes:
        cfg.update(change)
        started=time.perf_counter();current=session.update(cfg);elapsed=time.perf_counter()-started
        egg_preview.preview_files=reference_module.preview_files
        try: packed=prepare(raw,cfg)
        finally: egg_preview.preview_files=updated_preview
        n=struct.unpack('<Q',packed[:8])[0];header=json.loads(packed[8:8+n]);offset=8+n;files={}
        for file in header['files']:
            files[file['name']]=packed[offset:offset+file['size_bytes']];offset+=file['size_bytes']
        reference=json.loads(files['egg.ptb.json'])
        assert json.loads(json.dumps(current['preview']))==reference['preview'], change
        assert base64.b64decode(current['psd_base64'])==files['egg_PSD.png'], change
        if current['audio_base64']: assert base64.b64decode(current['audio_base64'])==files['egg_AUDIO.wav']
        records.append(dict(change=change,seconds=round(elapsed,4)))
        print(json.dumps(records[-1]),flush=True)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before
    out=ROOT/'output/m03-r2';out.mkdir(parents=True,exist_ok=True)
    (out/'natural-parity.json').write_text(json.dumps(dict(sha256=before,checks=records),indent=2),encoding='utf-8')


if __name__=='__main__': main()
