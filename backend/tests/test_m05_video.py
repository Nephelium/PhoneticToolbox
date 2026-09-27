import io
import json
from pathlib import Path
import sys
import pytest
import gzip
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'backend/src'),str(ROOT/'packages/phonetic_core/src')]
from ptb_worker.m05_video import JsonLinesSink, VideoLimits, analyze_video
from phonetic_core.lip.sequence import LipConfig

def test_rotation_pixels_match_independent_original_opencv(monkeypatch):
    import hashlib
    import mediapipe as mp
    from types import SimpleNamespace
    expected=json.loads((ROOT/'tests/fixtures/m05/rotation.json').read_text('utf8'))
    source=ROOT/expected['input'];assert hashlib.sha256(source.read_bytes()).hexdigest()==expected['input_sha256']
    seen=[]
    class Probe:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def process(self,image):
            if not seen:seen.append((list(image.shape),hashlib.sha256(image.tobytes()).hexdigest()))
            return SimpleNamespace(multi_face_landmarks=None)
    monkeypatch.setattr(mp.solutions.face_mesh,'FaceMesh',Probe)
    sink=io.BytesIO();meta=analyze_video(source,JsonLinesSink(sink,5_000_000),LipConfig(False))
    assert seen==[(expected['shape'],expected['rgb_sha256'])]
    assert meta['coordinates']['display_rotations']==[90]


def test_budget_and_short_write():
    sink=JsonLinesSink(io.BytesIO(),4)
    with pytest.raises(ValueError,match='output_budget'):sink.write({'x':3})
    class Short:
        def write(self,value):return 1
    with pytest.raises(OSError,match='short_result'):JsonLinesSink(Short(),100).write({'x':3})


def test_actual_vfr_and_cancel():
    source=ROOT/'output/validation/m05/inputs/motion-occlusion-vfr/input.mkv'
    manifest=json.loads((source.parent.parent/'manifest.json').read_text('utf-8'))
    expected=next(c for c in manifest['cases'] if c['variable_rate'])
    stream=io.BytesIO()
    result=analyze_video(source,JsonLinesSink(stream,5_000_000),LipConfig(False))
    rows=[json.loads(line) for line in stream.getvalue().splitlines()]
    assert [r['time_s'] for r in rows]==[r['time_s'] for r in expected['frames']]
    assert len(rows)==30 and result['timing']['decoded_frames']==30
    assert rows[0]['detected'] is False and rows[0]['metrics'] is None
    assert result['validity']['imputed_measurements']==0
    assert result['timing']['capture_fps'] is None
    with pytest.raises(InterruptedError):analyze_video(source,JsonLinesSink(io.BytesIO(),5_000_000),LipConfig(),stop=lambda:True)
    with pytest.raises(ValueError,match='resolution_budget'):analyze_video(source,JsonLinesSink(io.BytesIO(),5_000_000),LipConfig(),limits=VideoLimits(max_pixels=100))
    with pytest.raises(ValueError,match='frame_budget'):analyze_video(source,JsonLinesSink(io.BytesIO(),5_000_000),LipConfig(),limits=VideoLimits(max_frames=2))

@pytest.mark.parametrize('case',['front','motion-occlusion-vfr','resolution'])
def test_actual_migrated_legacy_detector_matches_independent_v2(case):
    with gzip.open(ROOT/'tests/fixtures/m05/v2.json.gz','rt',encoding='utf8') as stream:oracle=json.load(stream)
    expected=next(c for c in oracle['scientific'] if c['case']==case and not c['filter'])
    sink=io.BytesIO();analyze_video(ROOT/f'output/validation/m05/inputs/{case}/input.mkv',JsonLinesSink(sink,5_000_000),LipConfig(False))
    actual=[json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(actual)==len(expected['raw'])
    for i,row in enumerate(actual):
        assert (row['raw_points'] is None)==(expected['raw'][i] is None)
        if row['raw_points'] is not None:
            np.testing.assert_array_equal(row['raw_points'],expected['raw'][i])
            for key,value in row['metrics'].items():assert value==expected['metrics'][key][i]
