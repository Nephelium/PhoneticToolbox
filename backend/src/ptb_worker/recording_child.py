"""M16 headless operations; calls the same EDL/Praat/processing core as desktop."""
import io
import json
from pathlib import Path
import struct
import sys
import numpy as np
import soundfile as sf
from phonetic_core.recording import noise_profile, denoise, gain_audio, meter
from phonetic_core.recording.edits import select, remove, insert

MAX_AUDIO=100_000_000


def load(payload):
    if not 0<len(payload)<=MAX_AUDIO:raise ValueError('录音超过100 MB处理上限')
    info=sf.info(io.BytesIO(payload))
    if not 1<=info.channels<=8 or not 8000<=info.samplerate<=96000 or info.frames*info.channels*4+80>MAX_AUDIO:raise ValueError('录音格式或采样预算不支持')
    x,rate=sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
    if not len(x) or not np.isfinite(x).all():raise ValueError('录音为空或含非有限值')
    return x,rate


def compute(payloads,config):
    x,rate=load(payloads[0]);action=config['action'];roles=config['roles']
    if len(roles)!=x.shape[1]:raise ValueError('通道角色与录音不一致')
    start=config.get('start',0);end=config.get('end');end=len(x) if end is None else end
    spans=[{'source':0,'start':0,'end':len(x)}]
    part=select(spans,start,end)
    meta={'schema_version':'ptb-recording-processing/1','action':action,'sample_rate':rate,'roles':roles,'source_ids':['METHOD-M16-SPECTRAL-SUBTRACTION','SRC-PRAAT']}
    y=x
    if action in ('noise','denoise'):
        source,noise_rate=(load(payloads[-1]) if action=='denoise' else (x,rate))
        a=config.get('reference_start',0) if action=='denoise' else start
        b=config.get('reference_end') if action=='denoise' else end
        b=len(source) if b is None else b
        channel=config['channel'];reference_roles=config.get('reference_roles') or roles
        if noise_rate!=rate or not 0<=channel<source.shape[1] or len(reference_roles)!=source.shape[1] or reference_roles[channel]!='microphone':raise ValueError('噪声样本必须是同采样率的麦克风通道')
        if not 0<=a<b<=len(source) or not 3072<=b-a<=rate*60:raise ValueError('噪声样本需至少3072帧、最多60秒')
        profile,details=noise_profile(source[a:b,channel]);meta.update(details)
        if action=='noise':return None,meta
    if action in ('gain','denoise'):
        channels=config['channels']
        if not channels or any(roles[c]!='microphone' for c in channels):raise ValueError('请选择麦克风通道；EGG和其他通道保持不变')
        y=np.empty_like(x);block_frames=131072;halo=1024
        for a in range(0,len(x),block_frames):
            b=min(len(x),a+block_frames);left=max(0,a-halo);right=min(len(x),b+halo)
            result=denoise(x[left:right],profile,channels,config['strength']) if action=='denoise' else gain_audio(x[left:right],config['gain_db'],channels)
            y[a:b]=result[a-left:b-left]
        untouched=[c for c in range(x.shape[1]) if c not in channels]
        if untouched and not np.array_equal(x[:,untouched],y[:,untouched]):raise ValueError('未处理通道发生变化')
        meta.update(processed_channels=channels,algorithm='ptb-spectral-subtraction/1' if action=='denoise' else 'linear-gain/1',strength=config['strength'],gain_db=config['gain_db'],boundary='1024-frame halo; 131072-frame core blocks; zero pad at outer endpoints')
    elif action in ('keep','delete','cut','paste'):
        sources=[x]
        if action=='paste':
            clip,clip_rate=load(payloads[-1]);reference_roles=config.get('reference_roles') or roles
            if clip_rate!=rate or clip.shape[1]!=x.shape[1] or reference_roles!=roles:raise ValueError('仅允许同采样率、通道结构和角色粘贴')
            sources.append(clip)
            a=config.get('reference_start',0);b=config.get('reference_end');b=len(clip) if b is None else b
            clipboard=select([{'source':1,'start':0,'end':len(clip)}],a,b)
            edited=insert(spans,start,clipboard)
        else:edited=part if action=='keep' else remove(spans,start,end)
        count=sum(s['end']-s['start'] for s in edited)
        if not 0<count or count*x.shape[1]*4+80>MAX_AUDIO:raise ValueError('操作产生空录音或超过100 MB，当前版本保留')
        y=np.concatenate([sources[s['source']][s['start']:s['end']] for s in edited]);meta.update(operation=action,range=[start,end])
    elif action=='export':
        if start==end:raise ValueError('导出选区为空')
        y=x[start:end]
    if not np.isfinite(y).all():raise ValueError('处理产生非有限值')
    subtype=config['subtype'] if action=='export' else 'FLOAT'
    if subtype!='FLOAT':
        limit=1-2**(-15 if subtype=='PCM_16' else -23)
        if np.max(y,initial=0)>limit or np.min(y,initial=0)<-1:raise ValueError('PCM导出超过量化范围，请选float32或先显式减小增益')
    stream=io.BytesIO();sf.write(stream,y,rate,format='WAV',subtype=subtype);raw=stream.getvalue()
    if len(raw)>MAX_AUDIO:raise ValueError('输出超过100 MB')
    info=sf.info(io.BytesIO(raw))
    if info.frames!=len(y) or info.samplerate!=rate or info.channels!=x.shape[1] or info.subtype!=subtype:raise ValueError('输出回读格式不一致')
    meta.update(frames=len(y),channels=x.shape[1],subtype=subtype,quality=meter(y))
    return raw,meta


def main():
    request=json.loads(Path(sys.argv[1]).read_text('utf-8'))
    try:
        payloads=[Path(p).read_bytes() for p in request['paths']]
        wav,meta=compute(payloads,request['config']);header=json.dumps({'ok':True,'metadata':meta},ensure_ascii=False,allow_nan=False).encode()
    except Exception as exc:
        wav=None;header=json.dumps({'ok':False,'error':str(exc)[:400]},ensure_ascii=False).encode()
    with open(sys.argv[2],'wb') as output:output.write(struct.pack('<I',len(header))+header+(wav or b''))


if __name__=='__main__':main()
