"""M11-R1 native failure gates and output equality; public fixtures only."""
import json
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4
from ptb_worker.mfa.probe import generate
from ptb_worker.mfa.runtime import run

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'output/validation/m11-r1'/('native-'+uuid4().hex);out.mkdir(parents=True);print(out,flush=True)
    registry=json.loads((ROOT/'output/validation/m11/qt-642b638d008c497abb576e5db63dcbe7/components/registry.json').read_text('utf8'))
    runtime=registry['runtimes'][0]
    report={'success':False,'cases':[],'scope':'actual bounded Windows MFA 3.3.8; public synthetic cases and independent saved-result comparisons'}
    def case(name,expected,*,dictionary='啊\ta˥˥\n',text='啊 啊 啊 啊',tiers=None,cancel=False,timeout=180):
        root=out/name;generate(root/'corpus',word='啊')
        (root/'corpus/probe.lab').write_text(text,encoding='utf8')
        if tiers is not None:
            (root/'corpus/probe.lab').rename(root/'source.lab')
            content='File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 2.8\ntiers? <exists>\nsize = '+str(len(tiers))+'\nitem []:\n'
            for i,(name_,text_) in enumerate(tiers,1):
                content+=f'    item [{i}]:\n        class = "IntervalTier"\n        name = "{name_}"\n        xmin = 0\n        xmax = 2.8\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 2.8\n            text = "{text_}"\n'
            (root/'corpus/probe.TextGrid').write_text(content,encoding='utf8')
        shutil.copyfile(registry['models'][0]['model'],root/'model.zip')
        (root/'dictionary.dict').write_text(dictionary,encoding='utf8')
        resources={};started=time.monotonic()
        try:
            result=run(runtime['path'],root,dict(action='align',model=str(root/'model.zip'),dictionary=str(root/'dictionary.dict'),config=dict(beam=100,retry_beam=400),expected_files=1),
                       stop=lambda:cancel and time.monotonic()-started>1,timeout=timeout,evidence=resources)
            actual='success' if result['success'] else result.get('error')
        except Exception as exc:result=dict(success=False,error=str(exc));actual=str(exc)
        row=dict(name=name,result=result,resources=resources);report['cases'].append(row)
        (root/'report.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8')
        assert actual==expected,(name,actual,expected)
        assert resources['group_cleaned']
        if name=='oov':
            log=(root/'native.log').read_text('utf8');assert 'not_a_known_word' in log and 'Generating MFCC' not in log
        print(json.dumps(dict(name=name,outcome=actual,seconds=resources['elapsed_seconds'],group_cleaned=resources['group_cleaned'])),flush=True)
    try:
        case('oov','m11_oov_words',text='not_a_known_word')
        case('dictionary-phone-mismatch','m11_model_mismatch',dictionary='啊\tINVALID_PHONE\n')
        case('local-pinyin-phones-mismatch','m11_model_mismatch',dictionary=(Path.home()/'Documents/MFA/pretrained_models/dictionary/mandarin_pinyin_tab.dict').read_text('utf8'),text='dai1 dai1 tai2 tai2 tai1 tai1')
        case('phones-only','m11_transcript_tiers',tiers=[('phones','a')])
        case('ambiguous-words','m11_transcript_tiers',tiers=[('speaker1 words','啊'),('speaker2 words','啊')])
        case('cancel','cancelled',cancel=True)
        case('timeout','m11_timeout',timeout=1)
        baseline=json.loads((ROOT/'output/validation/m11-r1/before-900a136a52504ea294b718e14ddc85ed/report.json').read_text('utf8'))
        comparisons=[]
        for label in ('threaded','cache-cold','cache-warm'):
            candidates=[p for p in (ROOT/'output/validation/m11-r1').glob(label+'-*/report.json') if not p.parent.name.startswith('threaded-oov')]
            assert len(candidates)==1,(label,candidates)
            value=json.loads(candidates[0].read_text('utf8'))
            assert value['result']['textgrids']==baseline['result']['textgrids']
            if 'fingerprint' in value:assert value['fingerprint']==baseline['fingerprint']
            comparisons.append(dict(label=label,exact_textgrids=True,seconds=value['elapsed_seconds']))
        report['comparisons']=comparisons;report['success']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(dict(success=report['success'],out=str(out))),flush=True)


if __name__=='__main__':main()
