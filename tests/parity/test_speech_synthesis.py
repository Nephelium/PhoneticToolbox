import json
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
from phonetic_core.synthesis.klatt.api import defaults,validate,generate,synthesize,extract,export_parameters,import_parameters
from phonetic_core.synthesis.klatt.engine import Engine
from phonetic_core.models.audio import AudioInput

FIX=Path(__file__).parents[1]/'fixtures/m06'
META=json.loads((FIX/'v2.json').read_text('utf8'))
DATA=np.load(FIX/'v2.npz')


def config(case):
    c=defaults()
    for key in ('duration','sequence','sample_rate','curves','silence','boundaries'):c[key]=deepcopy(case[key])
    return c


@pytest.mark.parametrize('i',range(6))
def test_independent_v2_synthesis(i):
    case=META['cases'][i];c=config(case)
    for n,curve in Engine(c).params.items():np.testing.assert_array_equal(curve.get_array(c['duration'],c['sample_rate']),DATA[f'{i}_{n}'])
    np.random.seed(case['seed']);actual=synthesize(c)
    np.testing.assert_array_equal(actual,DATA[f'{i}_audio'])


@pytest.mark.parametrize('i',range(1,6))
def test_independent_v2_vowels(i):
    c=config(META['cases'][i]);c['curves']=defaults()['curves']
    for curve in c['curves'].values():curve['points'][-1][0]=c['duration']
    result=generate(c);expected=META['cases'][i]
    assert result['silence']==expected['silence']
    assert result['boundaries']==expected['boundaries']
    for n in ('F1','F2','F3','F4','F5','AV'):assert result['curves'][n]['points']==expected['curves'][n]['points']


def test_independent_extraction():
    from scipy.io import wavfile
    rate,samples=wavfile.read(FIX/'source.wav');result=extract(defaults(),AudioInput(samples,rate))
    for n,curve in result['curves'].items():np.testing.assert_array_equal(curve['points'],DATA['extracted_'+n],err_msg=n)


@pytest.mark.parametrize('text',['','p','a?i','*a','a\ni'])
def test_bad_ipa(text):
    c=defaults();c['sequence']=text
    with pytest.raises(ValueError,match='m06_'):generate(c)


@pytest.mark.parametrize('key,value',[('duration',float('nan')),('duration',0),('smooth',0),('fade_in',-1),('sample_rate',0),('f0_range',[500,50])])
def test_invalid_config(key,value):
    c=defaults();c[key]=value
    with pytest.raises(ValueError,match='m06_'):validate(c)


def test_complete_roundtrip_and_legacy():
    c=config(META['cases'][3]);c['curves']['F0']['points']=[[0.,100.],[c['duration'],180.]];c['curves']['F0']['override']=150.
    assert import_parameters(export_parameters(c))==c
    assert import_parameters(json.dumps(c))==c
    assert import_parameters(json.dumps(dict(schema_version="m06/1",config=c,seed=123,action="synthesize")))==c
    old='Parameter,Time,Value,Global\n__DURATION__,1,0,False\nF0,0,140,True\nShimmer,0,0.03,True\n'
    v=import_parameters(old);assert v['duration']==1 and v['curves']['Shimmer']['override']==.03
    with pytest.raises(ValueError):import_parameters('Parameter,Time,Value,Global\nUnknown,0,1,True')


def test_cancel_no_mutation():
    c=defaults();before=json.dumps(c)
    with pytest.raises(InterruptedError):synthesize(c,cancelled=lambda:True)
    assert before==json.dumps(c)


def test_core_has_no_ui_device_import():
    import ast
    from phonetic_core.synthesis.klatt import api
    for file in Path(api.__file__).parent.glob('*.py'):
        tree=ast.parse(file.read_text('utf8'))
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        assert not any(x.startswith(('PyQt','sounddevice','fastapi','sqlite','http')) for x in imports)


def test_pcm16_matches_v2_soundfile():
    import io,soundfile as sf,hashlib
    from ptb_worker.m06_child import compute
    for i in range(6):
        a=DATA[f'{i}_audio'].astype(np.float32);stream=io.BytesIO();sf.write(stream,a,META['cases'][i]['sample_rate'],format='WAV');stream.seek(0)
        expected,_=sf.read(stream,dtype='int16')
        values=dict(compute(dict(parameters=export_parameters(config(META['cases'][i])),action='synthesize',seed=META['cases'][i]['seed'],input_sha256=hashlib.sha256(b'').hexdigest()),b''))
        actual,_=sf.read(io.BytesIO(values['synthesis.wav']),dtype='int16')
        np.testing.assert_array_equal(actual,expected)


def test_validation_import_is_lightweight():
    import subprocess
    result=subprocess.run([sys.executable,'-B','-c',"from phonetic_core.synthesis.klatt.api import defaults,validate;validate(defaults());import sys;assert 'numpy' not in sys.modules;assert 'parselmouth' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
