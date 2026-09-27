import numpy as np
import parselmouth
import pytest
from ptb_worker.m08_jobs import execute, read_authorized
from ptb_api.m08_models import M08Config


def sound():
    t=np.arange(16000)/16000
    return parselmouth.Sound(.3*np.sin(2*np.pi*150*t),16000)


def test_preview_and_synthesis_snapshot():
    snd=sound(); preview=execute(snd,dict(action='preview'),'public')
    emitted=[]
    out=execute(snd,dict(action='synthesize',start=.1,end=.8,modified_f0=preview['original_f0']),
        'public',emit=lambda n,s,m:emitted.append((n,s,m)))
    assert len(out['outputs'])==1
    name,result,meta=emitted[0]
    assert name=='public_0.10_0.80_modified_1.wav'
    assert meta['source_start_s']==.1 and meta['source_end_s']==.8
    assert result.n_samples==11200


def test_fixed_hertz_after_ratio():
    result=[]
    execute(sound(),dict(action='transform',pitch_ratio=1.2,pitch_hz=20),'public',emit=lambda n,s,m:result.append(s))
    pitch=result[0].to_pitch().selected_array['frequency']
    # Independent known 150-Hz tone: 150 * 1.2 + 20 = 200 Hz.
    assert abs(np.median(pitch[pitch>0])-200)<1


@pytest.mark.parametrize('config',[dict(action='transform',speed=0),dict(action='transform',speed=float('inf')),dict(action='linear'),dict(action='synthesize',end=1,modified_f0=[float('nan')])])
def test_invalid_wire(config):
    with pytest.raises(ValueError):M08Config.model_validate(config)


def test_cancel_before_compute():
    with pytest.raises(InterruptedError):execute(sound(),dict(action='preview'),'public',cancelled=lambda:True)


def test_cancel_between_outputs():
    emitted=[]
    cfg=dict(action='linear',start=0,end=1,points=[dict(time=.1,freqs=[100,150],mode='full'),dict(time=.8,freqs=[150],mode='constant')])
    with pytest.raises(InterruptedError):execute(sound(),cfg,'public',cancelled=lambda:bool(emitted),emit=lambda *x:emitted.append(x))
    assert len(emitted)==1


def test_write_failure_not_success():
    def fail(*args):raise OSError('quota_writer_failed')
    with pytest.raises(OSError,match='quota_writer'):execute(sound(),dict(action='transform'),'public',emit=fail)


@pytest.mark.parametrize('config,code',[(dict(action='transform',start=.2,end=.9),'requires_whole'),(dict(action='transform',speed=.00001),'output_budget'),(dict(action='synthesize',end=1,modified_f0=[100]),'curve_length')])
def test_range_curve_budget(config,code):
    with pytest.raises(ValueError,match=code):execute(sound(),config,'public',emit=lambda *args:None)


def test_input_hash(tmp_path):
    f=tmp_path/'public.wav';sound().save(str(f),'WAV')
    with pytest.raises(ValueError,match='changed'):read_authorized(f,'0'*64)
