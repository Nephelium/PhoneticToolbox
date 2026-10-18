"""Actual native PCM and control landmarks for signed VOT."""
import json
from pathlib import Path
import numpy as np
import pytest
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.animation import prepare_animation
from phonetic_core.vocal_tract.trajectory import validate_frames,sample_trajectory
from phonetic_core.vocal_tract.source import validate_source

ROOT=Path(__file__).resolve().parents[2]

def frames(engine,vot,closure=.15,release=.008,aspiration=0):
    base=json.loads((ROOT/'frontend/public/vocal-tract/default-presets.json').read_text('utf-8'))['presets']['t'] if engine.speaker=='JD2' else {'params':engine.presets['t']}
    a=dict(params=engine.presets['a'],name='a',lip_width=1.,f0=120.,duration=.2,source=validate_source(None))
    stop={**base,'name':'t','lip_width':1.,'f0':120.,'source':validate_source(None),'duration':closure+release+aspiration,
          'stop_timing':dict(closure=closure,release=release,aspiration=aspiration,vot=vot)}
    return validate_frames(engine,[stop,a])

@pytest.fixture(params=['JD2','W02'])
def engine(request):
    e=Engine(request.param,resource_dir=ROOT/'resources/vocal_tract/native')
    yield e
    e.close()

def test_negative_vot_has_native_periodic_prevoicing_and_preserves_closure(engine):
    negative=frames(engine,-.1)
    positive=frames(engine,.008)
    original=json.dumps(negative)
    y=prepare_animation(engine,negative,pictures_enabled=False)['audio']
    unvoiced=prepare_animation(engine,positive,pictures_enabled=False)['audio']
    start,end=round(.075*engine.sr),round(.14*engine.sr)
    x=y[start:end];noise=unvoiced[start:end];period=round(engine.sr/120)
    correlation=np.corrcoef(x[:-period],x[period:])[0,1]
    assert correlation>.8,correlation
    assert np.sqrt(np.mean(x*x))>10*np.sqrt(np.mean(noise*noise))
    assert np.sqrt(np.mean(x*x))>.05*np.sqrt(np.mean(y[round(.21*engine.sr):round(.3*engine.sr)]**2))
    assert sample_trajectory(negative,.02)['source']['opening_mm']==0
    assert sample_trajectory(negative,.02)['source']['vibration']==0
    assert sample_trajectory(negative,.1)['params']==negative[0]['params']
    assert sample_trajectory(negative,.1)['source']['vibration']==1
    assert len(y)==round(.358*engine.sr)
    assert json.dumps(negative)==original

def test_positive_onset_at_frame_end_does_not_voice_the_preceding_native_block(engine,monkeypatch):
    f=frames(engine,.05,aspiration=.042)
    assert sample_trajectory(f,.2)['source']['vibration']==pytest.approx(0,abs=1e-12)
    assert sample_trajectory(f,.201)['source']['vibration']==pytest.approx(.5)
    assert sample_trajectory(f,.202)['source']['vibration']==1
    calls=[];offset=0;block=engine.block_tube
    def record(tube,g,n=960):
        nonlocal offset
        calls.append((offset,offset+n,float(g[engine.gnames.index('rel_amp')])));offset+=n
        return block(tube,g,n)
    monkeypatch.setattr(engine,'block_tube',record)
    clip=prepare_animation(engine,f,pictures_enabled=False)
    onset=round(.2*engine.sr)
    assert all(v<=1e-10 for a,b,v in calls if b<=onset)
    assert any(v==1 and b==onset+96 for a,b,v in calls)
    assert len(clip['audio'])==round(.4*engine.sr)

def test_minimum_timing_and_continuous_release_landmarks(engine):
    f=frames(engine,-.01,closure=.01,release=.001)
    assert f[0]['duration']==pytest.approx(.011)
    assert sample_trajectory(f,.009)['params']==f[0]['params']
    np.testing.assert_allclose(sample_trajectory(f,.011)['params'],f[1]['params'],rtol=0,atol=1e-12)
    assert sample_trajectory(f,.009)['source']['vibration']==1
    y=prepare_animation(engine,f,pictures_enabled=False)['audio']
    assert len(y)==round(.211*engine.sr) and np.isfinite(y).all()

def test_stop_start_splits_native_blocks_without_early_prephonatory_adduction(engine,monkeypatch):
    f=frames(engine,-.1);lead={**f[1],'duration':.203};f=[lead,*f]
    calls=[];offset=0;block=engine.block_tube
    def record(tube,g,n=960):
        nonlocal offset
        calls.append((offset,offset+n,float(g[engine.gnames.index('rel_amp')])));offset+=n
        return block(tube,g,n)
    monkeypatch.setattr(engine,'block_tube',record)
    prepare_animation(engine,f,pictures_enabled=False)
    boundary=round(.203*engine.sr)
    assert any(b==boundary and a<b and v==1 for a,b,v in calls)
    assert any(a==b==boundary and v==0 for a,b,v in calls)
    assert not any(a<boundary<b for a,b,v in calls)

def test_old_timing_documents_and_packs_remain_readable(engine):
    from phonetic_core.vocal_tract.document import make_document,sequence_document
    from phonetic_core.vocal_tract.preset_pack import make_pack,validate_pack
    f=frames(engine,-.1,aspiration=.042)
    doc=make_document(engine,f,[]);assert doc['stop_timing_method']=='m10/stop-timing/2'
    old={**doc,'stop_timing_method':'m10/stop-timing/1'}
    restored=sequence_document(engine,old);assert restored['frames']==doc['frames']
    assert '前浊' in restored['geometry_notice']
    pack=make_pack(engine,{'presets':[f[0]]});assert pack['stop_timing_method']=='m10/stop-timing/2'
    pack.pop('stop_timing_method');assert '前浊' in validate_pack(engine,pack)['geometry_notice']
    with pytest.raises(ValueError):sequence_document(engine,{**doc,'stop_timing_method':'future'})
