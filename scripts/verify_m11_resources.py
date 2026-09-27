"""Real external MFA behaviour and resource gates on public deterministic inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4
from ptb_worker.mfa.probe import generate
from ptb_worker.mfa.runtime import run
from ptb_worker.mfa.components import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--model',required=True);p.add_argument('--skip-repeat',action='store_true');p.add_argument('--skip-batch',action='store_true');p.add_argument('--errors-only',action='store_true');p.add_argument('--faults-only',action='store_true');a=p.parse_args()
    out=Path(__file__).resolve().parents[1]/'output/validation/m11'/('resources-'+uuid4().hex);out.mkdir(parents=True);print(out,flush=True)
    cases=[];report=dict(success=False,cases=cases,scope='Windows MFA 3.3.8 public synthetic inputs; no scientific accuracy claim')
    def case(name,syllables=4,count=1,text=None,dictionary='a\ta˥˥\n',timeout=180,cancel_after=None,bad_model=False,textgrid=False,crash=False):
        root=out/name;root.mkdir();generate(root/'corpus',syllables=syllables)
        if text is not None:(root/'corpus/probe.lab').write_text(text,encoding='utf8')
        if textgrid:
            # Isolated source fixture, renamed rather than deleting.
            (root/'corpus/probe.lab').rename(root/'source-transcript.lab')
            (root/'corpus/probe.TextGrid').write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 2.8\ntiers? <exists>\nsize = 1\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "utterance"\n        xmin = 0\n        xmax = 2.8\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 2.8\n            text = "a a a a"\n',encoding='utf8')
        for index in range(1,count):
            shutil.copyfile(root/'corpus/probe.wav',root/f'corpus/probe{index}.wav');shutil.copyfile(root/'corpus/probe.lab',root/f'corpus/probe{index}.lab')
        shutil.copyfile(a.model,root/'model.zip')
        if bad_model:(root/'model.zip').write_bytes(b'invalid public model fixture')
        (root/'dictionary.dict').write_text(dictionary,encoding='utf8');resources={};started=time.monotonic()
        from contextlib import nullcontext
        from unittest.mock import patch
        from ptb_worker.native import windows
        original=windows.OwnedProcess
        class CrashingProcess(original):
            def poll(self):
                if time.monotonic()-started>2:windows.terminate_process(self.info.process,77)
                return super().poll()
        try:
            with patch.object(windows,'OwnedProcess',CrashingProcess) if crash else nullcontext():
                result=run(a.runtime,root,dict(action='align',model=str(root/'model.zip'),dictionary=str(root/'dictionary.dict'),config=dict(beam=10,retry_beam=40),expected_files=count),
                timeout=timeout,stop=lambda:cancel_after is not None and time.monotonic()-started>cancel_after,evidence=resources)
        except Exception as exc:result=dict(success=False,error=str(exc))
        row=dict(name=name,result=result,resources=resources);atomic_json(root/'report.json',row)
        print(json.dumps(dict(name=name,success=result['success'],error=result.get('error'),resources=resources)),flush=True)
        return row
    try:
        if not a.skip_repeat:
            # Concurrent independent roots exercise real caches, SQLite and output isolation.
            with ThreadPoolExecutor(2) as pool:
                runs=[pool.submit(case,'repeat-'+str(i)) for i in (1,2)]
                first,second=[f.result() for f in runs]
            cases.extend([first,second]);assert first['result']['success'] and second['result']['success']
            assert first['result']['textgrids']==second['result']['textgrids'],'nondeterministic TextGrid'
        successes=[('textgrid-transcript',dict(textgrid=True))]
        if not a.skip_batch:successes.insert(0,('100-file-boundary',dict(count=100)))
        for name,kwargs in ([] if a.errors_only else successes):
            row=case(name,**kwargs);cases.append(row);assert row['result']['success'],row['result']
        for name,kwargs,expected in [('long-unsegmented-119.8s',dict(syllables=199),'m11_no_alignments'),('oov',dict(text='not_a_known_word'), 'm11_oov_words'),
                ('dictionary-mismatch',dict(dictionary='a\tIMPOSSIBLEPHONE\n'),'m11_model_mismatch'),
                ('model-corrupt',dict(bad_model=True),'m11_model_mismatch'),
                ('crash',dict(crash=True),'m11_process_crashed'),('cancel',dict(cancel_after=2),'cancelled'),('timeout',dict(timeout=2),'m11_timeout')]:
            if a.errors_only and name in ('long-unsegmented-119.8s','oov'):continue
            if a.faults_only and name=='dictionary-mismatch':continue
            row=case(name,**kwargs);cases.append(row);assert not row['result']['success'] and row['result']['error']==expected,row['result']
        assert all(c['resources']['group_cleaned'] for c in cases)
        report['success']=True
    finally:atomic_json(out/'report.json',report)


if __name__=='__main__':main()
