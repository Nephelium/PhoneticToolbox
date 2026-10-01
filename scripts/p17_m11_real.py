"""P17 MFA: exact natural WAV/LAB with existing optional runtime/model/dictionary.

The dictionary/model compatibility result is evidence, never assumed success.
No generated audio, no installation, no existing task database.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4
from phonetic_core.transcription.mfa_name_codec import encode_fs_name
from ptb_worker.mfa.runtime import run

ROOT=Path(__file__).resolve().parents[1]

def main():
    audio_root=Path('C:/Users/13680/Desktop/project/音频数据')
    lab=next(p for p in sorted(audio_root.rglob('*.lab')) if p.with_suffix('.wav').is_file())
    registry=json.loads((ROOT/'output/validation/m11/qt-642b638d008c497abb576e5db63dcbe7/components/registry.json').read_text('utf8'))
    runtime=registry['runtimes'][0]['path'];model=Path(registry['models'][0]['model'])
    dictionary=Path(registry['models'][0]['dictionary']).with_name('mandarin_pinyin_tab.dict')
    out=ROOT/'output/validation/p17/M11'/uuid4().hex;corpus=out/'corpus';corpus.mkdir(parents=True)
    originals=[]
    for p in [lab,lab.with_suffix('.wav')]:
        originals.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        shutil.copyfile(p,corpus/encode_fs_name(p.name))
    shutil.copyfile(dictionary,out/'dictionary.dict');shutil.copyfile(model,out/'acoustic.zip')
    report={'success':False,'runtime':runtime,'model':str(model),'dictionary':str(dictionary),'originals':originals,'scope':'natural recording runtime probe; alignment accuracy and UI not implied'}
    evidence={};started=time.perf_counter()
    try:report['result']=run(runtime,out,{'action':'align','model':str(out/'acoustic.zip'),'dictionary':str(out/'dictionary.dict'),'config':{'beam':10,'retry_beam':40},'expected_files':1},evidence=evidence);report['success']=bool(report['result'].get('success'))
    except Exception as exc:report['error']=str(exc)
    report.update(elapsed=time.perf_counter()-started,resources=evidence,originals_unchanged=all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in originals))
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(json.dumps({'out':str(out),'success':report['success'],'error':report.get('error'),'result':report.get('result'),'elapsed':report['elapsed']},ensure_ascii=False))

if __name__=='__main__':main()
