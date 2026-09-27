"""M08 frozen V2 numerical gate: exact comparison, no widened tolerance."""
import json
from pathlib import Path
import numpy as np
import parselmouth
import pytest
from phonetic_core.manipulation.m08_synthesis import synthesize_from_pitch
from phonetic_core.manipulation.m08_transform import transform
from phonetic_core.manipulation.m08_batch import generate_batch_linear
from phonetic_core.manipulation.m08_rules import track, import_sequence, next_name, validate_controls

ROOT=Path(__file__).resolve().parents[1]/'fixtures/m08'
DATA=np.load(ROOT/'v2.npz'); META=json.loads((ROOT/'v2.json').read_text())


@pytest.fixture(autouse=True)
def seed():parselmouth.praat.run("random_initializeWithSeedUnsafelyButPredictably (42)")


def sound():return parselmouth.Sound(DATA['samples'],META['sample_rate'])


def test_pitch_frames():
    times,f0=track(sound())
    np.testing.assert_array_equal(times,DATA['times'])
    np.testing.assert_array_equal(f0,DATA['f0'])


@pytest.mark.parametrize('i',range(4))
def test_synthesis(i):
    start,end,speed=META['synth'][i]
    out=synthesize_from_pitch(sound(),DATA['times'],DATA['f0']*1.2,start,end,speed)
    np.testing.assert_array_equal(out.values,DATA[f'synth_{i}'])
    np.testing.assert_array_equal([out.xmin,out.xmax,out.dx,out.x1],DATA[f'axis_{i}'])


@pytest.mark.parametrize('i,args',list(enumerate([(1,1,0),(.8,1,0),(1,1.2,20),(1.5,.8,-10),(1.005,1.005,.005)])))
def test_transform(i,args,tmp_path):
    if f'transform_{i}' in META['errors']:
        with pytest.raises(parselmouth.PraatError,match='Unit'):
            transform(parselmouth.Sound(DATA['pcm_input'],META['sample_rate']),*args)
        return
    out=transform(parselmouth.Sound(DATA['pcm_input'],META['sample_rate']),*args)
    path=tmp_path/'actual.wav';out.save(str(path),'WAV')
    np.testing.assert_array_equal(parselmouth.Sound(str(path)).values,DATA[f'transform_{i}'])


@pytest.mark.parametrize('i',range(16))
def test_connections(i,tmp_path):
    case=META['batch'][i]
    outputs=list(generate_batch_linear(sound(),'public',DATA['times'],DATA['f0'],0,.6,**case['args']))
    assert sorted(n for n,_,_ in outputs)==case['names']
    for j,(name,out,_) in enumerate(sorted(outputs,key=lambda x:x[0])):
        path=tmp_path/name;out.save(str(path),'WAV')
        np.testing.assert_array_equal(parselmouth.Sound(str(path)).values,DATA[f'batch_{i}_{j}'])


def test_offset(tmp_path):
    outputs=list(generate_batch_linear(sound(),'public',DATA['times'],DATA['f0'],0,.6,.1,.5,[-20],[20],[],'constant','constant',[],True))
    path=tmp_path/'actual.wav';outputs[0][1].save(str(path),'WAV')
    np.testing.assert_array_equal(parselmouth.Sound(str(path)).values,DATA['offset'])


def test_import_view_and_gaps():
    t=np.arange(7,dtype=float);f=np.array([0,100,110,120,0,130,0.])
    actual=import_sequence(t,f,0,4,'99 200\n88 300')
    np.testing.assert_array_equal(actual,[0,200,250,300,0,130,0]);assert f[1]==100
    with pytest.raises(ValueError,match='discontinuous'):import_sequence(t,f,0,6,'200')
    with pytest.raises(ValueError,match='unvoiced'):import_sequence(t,f,0,0,'200')


def test_numbering():
    assert next_name('a',.1,.5,['a_0.10_0.50_modified_x_7.wav','a_0.10_0.60_modified_99.wav'])=='a_0.10_0.50_modified_8.wav'


def test_control_errors():
    with pytest.raises(ValueError,match='diagonal'):validate_controls([dict(time=0,freqs=[100],mode='order'),dict(time=1,freqs=[100,200],mode='reverse')],0,1)
    with pytest.raises(ValueError,match='budget'):validate_controls([dict(time=0,freqs=list(range(20)),mode='full'),dict(time=1,freqs=list(range(20)),mode='full')],0,1)
