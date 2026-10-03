"""Cross-platform readback of this task's selected outputs; no physical devices."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import wave

root=Path(__file__).resolve().parents[1]
codec=importlib.util.find_spec('av') is not None
checks=[]
for relative in sys.argv[1:]:
    saved=root/relative
    for directory in sorted(saved.glob('M05-recording-*')):
        receipt=json.loads((directory/'recording-export.json').read_text('utf8'))
        names={f['name'] for f in receipt['files']}
        assert {p.name for p in directory.iterdir()}==names|{'recording-export.json'}
        for item in receipt['files']:
            payload=(directory/item['name']).read_bytes()
            assert len(payload)==item['bytes'] and hashlib.sha256(payload).hexdigest()==item['sha256']
        with wave.open(str(directory/'audio_recording.wav'),'rb') as w:
            assert w.getnframes()==receipt['audio']['samples'] and w.getframerate()==receipt['audio']['sample_rate']
        if 'audio_recording.lip.json' in names:
            cap=json.loads((directory/'capture.m05-preview.json').read_text('utf8'))
            lip=json.loads((directory/'audio_recording.lip.json').read_text('utf8'))['data']
            assert lip['metadata']['lip_manual_offset']==cap['lip_manual_offset']
            for index,row in enumerate(cap['frames']):
                assert math.isclose(lip['relative_times']['values'][index],row['time_s']-receipt['audio']['first_decoded_pts_s'],abs_tol=1e-12)
                for key in ('area','outer_width','open','circularity'):
                    assert lip[key]['values'][index]==(row.get('metrics') or {}).get(key)
        if codec:
            import av
            if 'raw_recording.mp4' in names:
                with av.open(str(directory/'raw_recording.mp4')) as c:times=[float(f.pts*f.time_base) for f in c.decode(video=0)]
                assert len(times)==receipt['video_frames']
                assert abs(times[0]-(receipt['source_first_video_pts_s']-receipt['mp4_origin_s']))<=1/90000
                assert all(a<b for a,b in zip(times,times[1:]))
            if 'face_animation.mp4' in names:
                with av.open(str(directory/'face_animation.mp4')) as c:
                    frames=list(c.decode(video=0));assert len(frames)==receipt['animation']['frames']
                with av.open(str(directory/'face_animation.mp4')) as c:
                    audio=list(c.decode(audio=0));assert audio
                    assert receipt['audio']['samples']<=sum(f.samples for f in audio)<=receipt['audio']['samples']+2048
        checks.append(dict(directory=str(directory.relative_to(root)),hashes=len(names),wav=True,lip='audio_recording.lip.json' in names,codec=codec))
assert checks
print(json.dumps(dict(success=True,platform=sys.platform,checks=checks,limits='Synthetic inputs; codec checks only when PyAV is present. No physical synchronization or Linux GUI claim.'),ensure_ascii=False,indent=2))
