"""Existing-frame export: actual codecs, full interior mesh and common clock."""
import json
import sys
import wave
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'backend/src'),str(ROOT/'packages/phonetic_core/src'),str(ROOT/'scripts')]
from verification_artifacts import verification_run

@pytest.fixture
def tmp_path():
    with verification_run('m05-offline','codec','Synthetic 478-point grid, 4 frames at 10fps, 1s 8kHz PCM silence; MP4/GIF readback') as (out,scratch):
        yield scratch

@pytest.fixture
def measured(tmp_path):
    # Distinct grid positions allow internal edges to be inspected independently
    # of face/lip contour rendering. Audio extends beyond measured video.
    points=[[10+(i%22)*2,10+(i//22)*2] for i in range(478)]
    rows=[dict(index=i,time_s=i*.1,detected=True,points=points,metrics={}) for i in range(4)]
    (tmp_path/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),'utf8')
    meta=dict(coordinates=dict(resolutions=[[64,64]]),timing=dict(anchor_s=0,last_video_pts_s=.3),audio=dict(first_decoded_pts_s=0))
    (tmp_path/'manifest.json').write_text(json.dumps(meta),'utf8')
    with wave.open(str(tmp_path/'audio_recording.wav'),'wb') as audio:
        audio.setparams((1,2,8000,0,'NONE','not compressed'));audio.writeframes(bytes(8000*2))
    return tmp_path,meta

def decoded(path):
    import av
    with av.open(str(path)) as media:
        sound=len(media.streams.audio)
        frames=[(float(f.pts*f.time_base),f.to_ndarray(format='rgb24')) for f in media.decode(video=0)]
    return frames,sound

def test_shared_preview_mesh_matches_frozen_mediapipe_edges():
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION,FACEMESH_CONTOURS
    spec=json.loads((ROOT/'resources/m05/metric-spec.json').read_text('utf8'))
    actual={(i,j) for i,neighbors in enumerate(spec['neighbors']) for j in neighbors if i<j}
    expected={tuple(sorted(edge)) for edge in FACEMESH_TESSELATION|FACEMESH_CONTOURS}
    assert actual==expected and len(actual)>1000

@pytest.mark.parametrize('offset',[.125,-.125])
def test_gif_and_mp4_keep_interior_mesh_same_frames_and_audio_extent(measured,tmp_path,offset):
    from ptb_worker.m05_media_export import result_media
    directory,meta=measured;infos={};outputs={}
    for format in ('mp4','gif'):
        target=tmp_path/format;target.mkdir()
        infos[format]=result_media(directory,target,options=dict(video=False,animation=True,quality='high',format=format),offset=offset,animation_name='animation.'+format)
        assert {p.name for p in target.iterdir()}=={'animation.'+format}
        frames,sound=decoded(target/('animation.'+format));outputs[format]=frames
        assert sound==(1 if format=='mp4' else 0)
        assert len(frames)==infos[format]['animation']['frames']
        assert frames[0][1].shape==(1080,1080,3)
        assert infos[format]['animation']['full_mesh'] and infos[format]['animation']['face_fitted']
    assert infos['gif']['animation']['frames']==infos['mp4']['animation']['frames']
    assert infos['gif']['animation']['output_origin_s']==infos['mp4']['animation']['output_origin_s']
    assert abs(outputs['gif'][-1][0]-outputs['mp4'][-1][0])<=.01
    # The centre has many interior mesh pixels, and the rasterized positions
    # agree despite GIF's palette and H.264's chroma quantization.
    index=6
    masks=[]
    for format in ('mp4','gif'):
        image=outputs[format][index][1].astype(int)
        mask=(image[:,:,1]>image[:,:,0]+30)&(image[:,:,1]>image[:,:,2]+30)
        assert np.count_nonzero(mask[300:780,300:780])>1000
        masks.append(mask)
    intersection=np.count_nonzero(masks[0]&masks[1]);union=np.count_nonzero(masks[0]|masks[1])
    assert intersection/union>.75
    # Audio tail retains matching blank pictures in both formats.
    assert np.min(outputs['gif'][-1][1])>=250 and np.min(outputs['mp4'][-1][1])>=250
