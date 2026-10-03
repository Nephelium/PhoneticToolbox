"""Local recording signal audit and optional method provenance, no physical devices."""
import sys
from pathlib import Path
import math
import json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'backend/src'),str(ROOT/'desktop/src'),str(ROOT/'packages/phonetic_core/src')]

@pytest.mark.parametrize('level',[0,5,8000])
def test_signal_audit_does_not_turn_silence_into_successful_voice(tmp_path,level):
    import wave
    from ptb_desktop.m05_capture import CaptureMux
    from ptb_worker.m05_media_export import recording_bundle
    source=tmp_path/'source.mkv';mux=CaptureMux(source,64,64,30,48000)
    samples=np.full((4800,1),level,np.int16)
    mux.audio_frame(samples,0)
    for i in range(3):mux.video_frame(np.zeros((64,64,3),np.uint8),round(i/30*1e9))
    mux.close();target=tmp_path/'saved';target.mkdir()
    (target/'capture.m05-preview.json').write_text(json.dumps(dict(backend='candidate-test',frames=[dict(time_s=0.,metrics=None),dict(time_s=.1,metrics=None)])),'utf8')
    info=recording_bundle(source,target)
    signal=info['audio']['signal']
    assert signal['low_signal']==(level<33)
    assert signal['peak_dbfs']==pytest.approx(20*math.log10(max(1e-6,level/32768)))
    with wave.open(str(target/'audio_recording.wav'),'rb') as w:
        np.testing.assert_array_equal(np.frombuffer(w.readframes(w.getnframes()),'<i2'),samples.ravel())
    assert not (target/'audio_recording.lip.json').exists()
    assert '不足两帧' in info['associated_exchange_note']
    assert (target/'candidate.lip.json').is_file()

def test_optional_method_provenance_survives_safe_reader_and_old_format():
    from ptb_worker.io.lip import encode_lip,decode_lip
    from ptb_worker.io.annotation import lip_preview
    from ptb_worker.io.limits import FormatError
    data=dict(relative_times=[0.,.1],open=[1.,2.],metadata={'source_backend':'mediapipe-web/0.10.14/float16-1/candidate','source_status':'candidate'})
    payload=encode_lip(data)
    assert decode_lip(payload)['metadata']==data['metadata']
    assert lip_preview(payload,'audio_recording.lip.json')['data']['metadata']==data['metadata']
    data['metadata']={};assert decode_lip(encode_lip(data))['metadata']=={}
    for bad in (123,'x'*129):
        data['metadata']={'source_backend':bad}
        with pytest.raises(FormatError):encode_lip(data)
