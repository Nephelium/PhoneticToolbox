"""M12 owned Chrome test RPC; real FileProvider/TaskBridge, no DB or EXE."""
import base64
import hashlib
import json
from pathlib import Path
import pickle
import sys
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from phonetic_core.annotation import parse_document
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.task_bridge import TaskBridge

ROOT = Path(__file__).resolve().parents[1]


def fixtures(out):
    inputs = out/'inputs'; inputs.mkdir()
    t = np.arange(32000)/16000
    samples = .4*np.sin(2*np.pi*180*t)*(((t>.23)&(t<.62))|((t>.83)&(t<1.18)))
    words=[(0,.2,''),(.2,.65,'ba1'),(.65,.8,''),(.8,1.2,'ba2'),(1.2,2,'')]
    phones=[(0,.2,''),(.2,.4,'b'),(.4,.65,'a1'),(.65,.8,''),(.8,1.,'b'),(1.,1.2,'a2'),(1.2,2,'')]
    def grid(word_intervals, phone_intervals):
        lines=['"ooTextFile short"','"TextGrid"','0','2','<exists>','3']
        for name, intervals in [('words',word_intervals),('phones',phone_intervals)]:
            lines.extend(['"IntervalTier"',f'"{name}"','0','2',str(len(intervals))])
            for a,b,label in intervals:lines.extend([str(a),str(b),f'"{label}"'])
        lines.extend(['"TextTier"','"events"','0','2','1','.5','"事件 ʔ"'])
        return '\n'.join(lines)+'\n'
    for name in ('audio_recording', 'second'):
        wavfile.write(inputs/(name+'.wav'),16000,samples.astype(np.float32))
        (inputs/(name+'.TextGrid')).write_text(grid(words,phones),encoding='utf-8')
    (inputs/'audio_recording.lab').write_text('ba1 ba2 ba3 ba4',encoding='utf-8')
    record={'relative_times':np.arange(60)/30,'open':np.sin(np.arange(60)/4)*.3+.5,'outer_width':np.cos(np.arange(60)/4)*.1+.7,
            'landmarks':np.arange(60*6).reshape(60,3,2).astype(float),'metadata':{'lip_manual_offset':.007,'session':'preserve'}}
    (inputs/'audio_recording.pkl').write_bytes(pickle.dumps(record,protocol=4))
    (out/'reference.TextGrid').write_text(grid([(0,2,'ref')],[(0,2,'ɹɛf')]),encoding='utf-8')
    (out/'custom.dict').write_text('井井 j i ŋ\nba1 p a1\nba2 p a2\n',encoding='utf-8')
    return inputs


def main():
    out=ROOT/'output/validation/m12-ui'/uuid4().hex;out.mkdir(parents=True)
    inputs=fixtures(out);provider=FileProvider();bridge=TaskBridge(provider,None)
    grant=provider.choose('input',lambda:inputs);natural=[]
    print(json.dumps({'ready':True,'out':str(out)},ensure_ascii=False),flush=True)
    for line in sys.stdin:
        req={}
        try:
            req=json.loads(line);op=req['op']
            if op=='shutdown':break
            if op=='choose':value=grant
            elif op=='natural':
                # Only the recordings already authorized and frozen for P03/M01.
                baseline=ROOT/'output/validation/p03/20260909-165413'
                for key in ('LOCAL-01','LOCAL-12'):
                    request=json.loads((baseline/key/'1/request.json').read_text('utf-8'))
                    wav=Path(request['input']);grid=Path(request['textgrid'])
                    import gzip
                    with gzip.open(baseline/key/'1/result.json.gz','rt',encoding='utf-8') as stream:old=json.load(stream)
                    assert hashlib.sha256(wav.read_bytes()).hexdigest()==old['input_sha256']
                    raw=grid.read_bytes();text=raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig');doc=parse_document(text)
                    natural.append({'case':key,'originals':[(p,hashlib.sha256(p.read_bytes()).hexdigest()) for p in (wav,grid)],'tiers':doc['tiers']})
                    (inputs/(key+'.wav')).write_bytes(wav.read_bytes());(inputs/(key+'.TextGrid')).write_bytes(raw)
                value=[{'case':n['case'],'word':n['tiers'][0]['name'],'phone':n['tiers'][1]['name'],'time':next((i['xmin']+i['xmax'])/2 for i in n['tiers'][0]['intervals'] if i['text'] and i['xmin']<2.8)} for n in natural]
            elif op=='natural_verify':
                value=[]
                for n in natural:
                    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in n['originals'])
                    result=parse_document((inputs/(n['case']+'_webedit.TextGrid')).read_text('utf-8'))
                    assert any(i['text']=='M12验证 æ' for i in result['tiers'][0]['intervals'])
                    value.append({'case':n['case'],'source_hashes_unchanged':True,'tiers':len(result['tiers']),'saved_text_verified':True})
                (out/'natural-report.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
            elif op=='read':
                raw,digest=provider.read(req['id']);value={'base64':base64.b64encode(raw).decode(),'sha256':digest}
            elif op=='inspect':
                value={'files':sorted(p.name for p in inputs.iterdir()),'lip_offset':pickle.loads((inputs/'audio_recording.pkl').read_bytes())['metadata']['lip_manual_offset'],
                       'grids':{p.name:parse_document(p.read_text('utf-8')) for p in inputs.glob('*.TextGrid')},
                       'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}}
            elif op=='external_change':
                p=inputs/'audio_recording_webedit.TextGrid';p.write_text(p.read_text('utf-8').replace('ba2','external'),encoding='utf-8');value=True
            else:value=bridge.invoke(req)
            print(json.dumps({'id':req['rpc_id'],'value':value},ensure_ascii=False),flush=True)
        except Exception as exc:
            print(json.dumps({'id':req.get('rpc_id'),'error':str(exc)},ensure_ascii=False),flush=True)
    provider.close()


if __name__=='__main__':main()
