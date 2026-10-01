"""P17 M08 real-input parity and cancellation. Original sources stay read-only."""
import argparse,hashlib,importlib.util,json,threading,time
from pathlib import Path
from uuid import uuid4
import numpy as np
import parselmouth
from ptb_worker.local_workspace import prepare_workspace
from ptb_worker.store import SQLiteJobStore,LOCAL_PROJECT
from ptb_worker.local_acoustic_files import LocalAcousticFiles
from ptb_worker.m08_task import submit
from ptb_worker.acoustic_executor import execute_acoustic_claim
from phonetic_core.manipulation.m08_synthesis import synthesize_from_pitch
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);raw=a.input.read_bytes();sha=hashlib.sha256(raw).hexdigest();checks=[]
 source=ROOT.parent/'PhoneticToolbox_v2/phonetic_toolbox/core/manipulation/synthesis.py';spec=importlib.util.spec_from_file_location('p17_v2_m08',source);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 sound=parselmouth.Sound(str(a.input));pitch=sound.to_pitch();times=pitch.xs();f0=pitch.selected_array['frequency'].copy();modified=f0.copy();modified[modified>0]*=1.1
 for speed in (.8,1,1.2):
  parselmouth.praat.run('random_initializeWithSeedUnsafelyButPredictably (42)');expected=legacy.synthesize_from_pitch(sound,times,modified,0,sound.xmax,speed);parselmouth.praat.run('random_initializeWithSeedUnsafelyButPredictably (42)');actual=synthesize_from_pitch(sound,times,modified,0,sound.xmax,speed);np.testing.assert_array_equal(expected.values,actual.values);assert expected.sampling_frequency==actual.sampling_frequency
  expected.save(str(a.out/f'v2-{speed}.wav'),'WAV');actual.save(str(a.out/f'v3-{speed}.wav'),'WAV');assert (a.out/f'v2-{speed}.wav').read_bytes()==(a.out/f'v3-{speed}.wav').read_bytes();checks.append(dict(speed=speed,array_values=int(actual.values.size),PCM16_byte_identical=True))
 db,cache=prepare_workspace(a.out/'workspace',ROOT/'backend/migrations');store=SQLiteJobStore(db);files=LocalAcousticFiles(store,cache);ref=files.import_input(raw,'真实录音.wav','audio')
 def run(cancel):
  job=submit(store,'local',dict(project_id=LOCAL_PROJECT,idempotency_key=uuid4().hex,audio=ref,config={'action':'preview'}));claim=store.claim('p17');stop=threading.Event();evidence={};t=time.perf_counter();execute_acoustic_claim(store,claim,'p17',stop,on_started=(lambda pid:stop.set()) if cancel else None,process_evidence=evidence);done=store.get('local',job['id']);assert done['state']==('cancelled' if cancel else 'succeeded'),done;assert evidence['cleaned'];return dict(state=done['state'],ms=(time.perf_counter()-t)*1000,evidence=evidence)
 cancel=run(True);recovery=run(False);assert hashlib.sha256(a.input.read_bytes()).hexdigest()==sha
 (a.out/'report.json').write_text(json.dumps(dict(success=True,input_sha256=sha,v2_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),parity=checks,cancel=cancel,recovery=recovery),indent=2),encoding='utf8');print(a.out)
if __name__=='__main__':main()
