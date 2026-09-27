"""Fixed M05 scientific child; receives a host-created request file only."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
if __name__=='__main__':sys.path[:0]=[str(ROOT/'backend/src'),str(ROOT/'packages/phonetic_core/src')]

def atomic(path,value):
    pending=path.with_suffix('.next')
    with pending.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    # A Windows reader can briefly hold the old status file without delete
    # sharing. Bounded retry preserves atomicity; never expose a partial JSON.
    for attempt in range(40):
        try:os.replace(pending,path);break
        except PermissionError:
            if attempt==39:raise
            time.sleep(.01)

def run(request,root):
    from ptb_worker.m05_video import analyze_video,JsonLinesSink,VideoLimits,sha256_file
    from ptb_worker.m05_results import export_tables
    from phonetic_core.lip.sequence import LipConfig
    config=request['config'];target=root/'results';target.mkdir()
    def progress(value):atomic(root/'status.json',value)
    with (target/'frames.jsonl').open('xb') as output:
        meta=analyze_video(root/request['input'],JsonLinesSink(output,400_000_000),
            LipConfig(config['filter_enabled'],config['cutoff_hz']),limits=VideoLimits(input_bytes=128_000_000,output_bytes=400_000_000),progress=progress)
        output.flush();os.fsync(output.fileno())
    meta['input_name']=request.get('original_name',request['input'])
    from ptb_worker.m05_audio import extract_audio
    meta['audio'],audio_names=extract_audio(root/request['input'],target)
    from ptb_worker.m05_alignment import suggest_offset
    meta['offset_suggestion']=suggest_offset(target,meta)
    names=['frames.jsonl',*audio_names,*export_tables(target/'frames.jsonl',target,meta)]
    if config.get('animation','none')!='none':
        from ptb_worker.m05_animation import export_animation
        name='lip-animation.'+config['animation']
        meta['animation']=export_animation(target/'frames.jsonl',target/name,meta,quality=config['quality'],format=config['animation'],offset=config['offset'],audio_path=target/'audio_recording.wav' if meta['audio']['present'] else None)
        names.append(name)
    meta['input_name']=request.get('original_name',request['input'])
    meta['files']=[dict(name=name,bytes=(target/name).stat().st_size,sha256=sha256_file(target/name)) for name in names]
    atomic(target/'manifest.json',meta);names.append('manifest.json')
    if sum((target/n).stat().st_size for n in names)>512_000_000:raise ValueError('m05_output_budget')
    return dict(success=True,files=names,frames=meta['timing']['decoded_frames'])

if __name__=='__main__':
    source=Path(sys.argv[1]);root=source.parent
    try:response=run(json.loads(source.read_text('utf-8')),root)
    except Exception as error:response=dict(success=False,error=(str(error) or type(error).__name__)[:100],type=type(error).__name__)
    atomic(root/'response.json',response)
