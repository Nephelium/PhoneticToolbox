"""Streaming readable M05 exports; measurement and legacy fill are separate tracks."""
import csv
import json
import math
from pathlib import Path
from .m05_video import sha256_file

KEYS=('area','face_width','face_height','height_px','outer_width_px','inner_width_px','total_width_px','open_px','length','height','outer_width','inner_width','total_width','open','circularity')


def rows(path):
    with Path(path).open('r',encoding='utf-8') as stream:
        for line in stream:
            if len(line)>200_000:raise ValueError('m05_frame_record_limit')
            yield json.loads(line)


def export_tables(frame_path, directory, metadata, stop=lambda:False):
    directory=Path(directory)
    total=metadata['timing']['decoded_frames']
    first=next((r for r in rows(frame_path) if r['detected']),None)
    preview=[];stride=max(1,math.ceil(total/240));last=first
    with (directory/'measurements.csv').open('x',newline='',encoding='utf-8-sig') as raw, (directory/'legacy-compatibility.csv').open('x',newline='',encoding='utf-8-sig') as compat:
        w=csv.writer(raw);c=csv.writer(compat)
        header=['frame_index','audio_relative_time_s','detected','imputed',*KEYS]
        w.writerow(header);c.writerow(header)
        for row in rows(frame_path):
            if stop():raise InterruptedError('m05_cancelled')
            if row['detected']:last=row
            observed=row['metrics'] or {}
            held=last['metrics'] if last else {}
            w.writerow([row['index'],row['time_s'],row['detected'],False,*[observed.get(k) for k in KEYS]])
            c.writerow([row['index'],row['time_s'],row['detected'],not row['detected'] and last is not None,*[held.get(k) for k in KEYS]])
            if row['index']%stride==0 or row['index']==total-1:
                preview.append({k:row[k] for k in ('index','time_s','detected','points','metrics','width','height') if k in row})
    # Four parameter tracks for the EXISTING inert ptb.lip/1 adapter, under its 2 MB budget.
    exchange=None
    if total<=6000:
        vectors={k:dict(values=[],nonfinite=[]) for k in ('relative_times','area','outer_width','open','circularity')}
        for row in rows(frame_path):
            if stop():raise InterruptedError('m05_cancelled')
            for key,vector in vectors.items():
                value=row['time_s'] if key=='relative_times' else (row['metrics'] or {}).get(key)
                vector['values'].append(value);vector['nonfinite'].append(0 if value is not None else 1)
        vectors['metadata']=dict(time_alignment_mode='anchored_audio_start',lip_manual_offset=0.,audio_first_frame_time=0.)
        encoded=json.dumps(dict(schema='ptb.lip/1',data=vectors),allow_nan=False,separators=(',',':')).encode()
        if len(encoded)<=2_000_000:
            exchange=directory/'audio_recording.lip.json';exchange.write_bytes(encoded)
    payload=dict(schema='m05-preview/1',backend=metadata['backend'],rows=preview,metadata=metadata,
                 display_stride=stride,display_only=True,full_frame_count=total,
                 legacy_compatibility='leading-first-value / last-value hold; never new measurements',
                 exchange_available=exchange is not None)
    (directory/'preview.json').write_text(json.dumps(payload,allow_nan=False,separators=(',',':')),encoding='utf-8')
    return ['measurements.csv','legacy-compatibility.csv','preview.json']+(['audio_recording.lip.json'] if exchange else [])


def save_offset_metadata(original, destination, offset, action):
    if action=='cancel':return False
    if action not in ('apply','save_without_offset') or not math.isfinite(offset) or abs(offset)>2:raise ValueError('invalid_offset')
    target=Path(destination)
    if target.exists():raise FileExistsError('Never overwrite original M05 results')
    data=json.loads(Path(original).read_text('utf-8'))
    data['timing']['lip_manual_offset']=offset if action=='apply' else 0.
    data['offset_application']='Add once to audio_relative_time_s; raw frames unchanged'
    data['parent_manifest_sha256']=sha256_file(original)
    with target.open('x',encoding='utf-8') as stream:json.dump(data,stream,ensure_ascii=False,allow_nan=False,indent=2)
    return True
