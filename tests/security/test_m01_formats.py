"""M01-C malformed allocation, inert deserialization and text behavior regressions."""
from dataclasses import replace
import io
import json
import pickle
import struct
from uuid import UUID
import numpy as np
import pytest
from scipy.io import wavfile
from ptb_worker.io.audio import decode_wav,slice_wav,encode_wav
from phonetic_core.models.audio import AudioInput
from ptb_worker.io.annotations import decode_textgrid
from ptb_worker.io.lip import (decode_lip,encode_lip,convert_local_legacy_lip,
    load_inert_legacy_pickle,legacy_companion_start)
from ptb_worker.io.limits import Limits,LimitError,FormatError
from phonetic_core.acoustic.lip import interpolate_lip,resolve_lip_time_axis


@pytest.mark.parametrize('dtype',['uint8','int16','int32','float32','float64'])
@pytest.mark.parametrize('channels',[1,2])
@pytest.mark.parametrize('rate',[16000,44100,48000])
def test_wav_original_scale_channels_and_rate(dtype,channels,rate):
    values=np.arange(30,dtype=dtype).reshape(-1,channels)
    if channels==1:values=values[:,0]
    stream=io.BytesIO();wavfile.write(stream,rate,values)
    result=decode_wav(stream.getvalue())
    assert result.sample_rate_hz==rate and result.samples.dtype==values.dtype
    np.testing.assert_array_equal(result.samples,values)


def test_wav_pcm24_is_left_justified_without_requantization():
    raw=b'\x00\x00\x80\xff\xff\x7f\x01\x00\x00'
    fmt=struct.pack('<HHIIHH',1,1,44100,132300,3,24)
    body=b'WAVEfmt '+struct.pack('<I',16)+fmt+b'data'+struct.pack('<I',9)+raw+b'\0'
    decoded=decode_wav(b'RIFF'+struct.pack('<I',len(body))+body)
    np.testing.assert_array_equal(decoded.samples,np.array([-2147483648,2147483392,256],dtype='int32'))


def test_extensible_pcm_guid_and_declared_extension_are_checked():
    fmt=struct.pack('<HHIIHHHHI',0xfffe,2,16000,64000,4,16,22,16,3)+UUID('00000001-0000-0010-8000-00aa00389b71').bytes_le
    def wav(f):
        body=b'WAVEfmt '+struct.pack('<I',len(f))+f+b'data'+struct.pack('<I',4)+struct.pack('<hh',-123,456)
        return b'RIFF'+struct.pack('<I',len(body))+body
    np.testing.assert_array_equal(decode_wav(wav(fmt)).samples,[[-123,456]])
    with pytest.raises(FormatError):decode_wav(wav(fmt[:16]+b'\xff\xff'+fmt[18:]))
    with pytest.raises(FormatError):decode_wav(wav(fmt[:-1]+b'\0'))


def test_wav_refuses_before_decoder(monkeypatch):
    stream=io.BytesIO();wavfile.write(stream,16000,np.zeros(100,dtype='int16'));payload=stream.getvalue()
    monkeypatch.setattr(wavfile,'read',lambda *a,**k:pytest.fail('Decoder ran before preflight'))
    with pytest.raises(LimitError):decode_wav(payload,replace(Limits(),samples=99))
    with pytest.raises(LimitError):decode_wav(payload,replace(Limits(),input_bytes=32))
    with pytest.raises(FormatError):decode_wav(payload[:-1])
    with pytest.raises(FormatError):decode_wav(payload[:40]+struct.pack('<I',0x7fffffff)+payload[44:])


def test_segment_preserves_integer_frame_rule_dtype_and_channels():
    data=np.arange(2000,dtype='int16').reshape(-1,2);audio=AudioInput(data,44100)
    segment=decode_wav(slice_wav(audio,.00101,.01003))
    np.testing.assert_array_equal(segment.samples,data[int(.00101*44100):int(.01003*44100)])
    assert segment.samples.dtype==data.dtype and segment.sample_rate_hz==44100
    for start,end in [(-1.,.001),(.1,.2),(.01,.001),(0.,1e-9)]:
        with pytest.raises(FormatError):slice_wav(audio,start,end)
    with pytest.raises(LimitError):encode_wav(audio,replace(Limits(),output_bytes=32))


LONG='''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1
tiers? <exists>
size = 1
item []:
item [1]:
 class = "IntervalTier"
 name = "音节 IPA"
 xmin = 0
 xmax = 1
 intervals: size = 2
 intervals [1]:
  xmin = 0
  xmax = 0.5
  text = "=literal əʊ"
 intervals [2]:
  xmin = 0.5
  xmax = 1
  text = "他说""啊""\n第二行"
'''
SHORT='''File type = "ooTextFile short"
"TextGrid"
0
1
<exists>
1
"IntervalTier"
"音节 IPA"
0
1
2
0
.5
"=literal əʊ"
.5
1
"他说""啊""\n第二行"
'''


@pytest.mark.parametrize('encoding',['utf-8','utf-8-sig','utf-16'])
def test_textgrid_short_long_chinese_ipa_quotes(encoding):
    a=decode_textgrid(LONG.encode(encoding));b=decode_textgrid(SHORT.encode(encoding))
    assert a==b and a[0].intervals[1].text=='他说"啊"\n第二行'
    assert a[0].intervals[0].text=='=literal əʊ'


@pytest.mark.parametrize('text',[LONG.replace('size = 1','size = 999999999',1),
    LONG.replace('xmax = 1','xmax = NaN',1),LONG.replace('xmin = 0.5','xmin = -1'),
    LONG.replace('IntervalTier','TextTier'),LONG.replace('size = 2','size = 3'),LONG[:-2]])
def test_textgrid_malformed_is_explicit(text):
    with pytest.raises(FormatError):decode_textgrid(text.encode())


def test_textgrid_limits():
    with pytest.raises(LimitError):decode_textgrid(LONG.encode(),replace(Limits(),text_bytes=64))
    with pytest.raises(FormatError):decode_textgrid(LONG.encode(),replace(Limits(),text_items=1))


def test_lip_safe_conversion_preserves_independent_axis_and_four_tracks():
    data={'absolute_timestamps':[100.5,100.,100.25,100.25,100.75,float('nan')],
          'relative_times':[10.5,10.,10.25,10.25,10.75,float('nan')],
          'metadata':{'audio_first_frame_time':100.,'lip_manual_offset':.125}}
    for i,key in enumerate(('area','outer_width','open','circularity'),1):data[key]=[3*i,i,2*i,99*i,4*i,5*i]
    data['area'][2]=float('nan')
    payload=convert_local_legacy_lip(pickle.dumps(data))
    assert b'nonfinite' in payload and b'null' in payload
    clean=decode_lip(payload)
    axis=resolve_lip_time_axis(clean,companion_start=99.)
    np.testing.assert_array_equal(axis.times,[0.,.25,.5,.75])
    assert axis.manual_offset==.125
    out=interpolate_lip(clean,np.array([0.,.125,.375,.625,.875,1.]))
    np.testing.assert_allclose(out['LipArea'],[np.nan,1.,2.,3.,4.,np.nan],equal_nan=True)
    for i,key in enumerate(('LipWidth','LipOpen','LipCirc'),2):
        np.testing.assert_allclose(out[key],[np.nan,i,2*i,3*i,4*i,np.nan],equal_nan=True)
    del clean['metadata']['audio_first_frame_time']
    assert resolve_lip_time_axis(clean,legacy_companion_start(pickle.dumps({'start_time':99.}))).times[0]==1.


def test_pickle_gadget_never_executes(tmp_path):
    target=tmp_path/'executed'
    class Attack:
        def __reduce__(self):return (eval,(f"__import__('pathlib').Path({str(target)!r}).write_text('bad')",))
    with pytest.raises(FormatError):load_inert_legacy_pickle(pickle.dumps(Attack()))
    assert not target.exists()
    with pytest.raises(FormatError):load_inert_legacy_pickle(pickle.dumps(np.array([1.])))


def test_pickle_memo_bomb_cycle_trailing_and_json_depth_rejected():
    with pytest.raises(LimitError):load_inert_legacy_pickle(b'\x80\x02]r\xff\xff\xff\x7f.')
    cycle=[];cycle.append(cycle)
    with pytest.raises(FormatError):load_inert_legacy_pickle(pickle.dumps(cycle))
    with pytest.raises(FormatError):load_inert_legacy_pickle(pickle.dumps({})+b'x')
    with pytest.raises(LimitError):decode_lip(b'['*10000+b']'*10000)
    with pytest.raises(FormatError):decode_lip(b'{"schema":"ptb.lip/1","schema":"ptb.lip/1","data":{}}')
    with pytest.raises(FormatError):decode_lip(b'{"schema":"ptb.lip/1","data":{"relative_times":[NaN,1]}}')


def test_lip_nonfinite_masks_and_offset_validation():
    data={'relative_times':[0.,1.,2.],'area':[float('nan'),float('inf'),float('-inf')]}
    clean=decode_lip(encode_lip(data))
    assert np.isnan(clean['area'][0]) and clean['area'][1:]==[float('inf'),float('-inf')]
    data['metadata']={'lip_manual_offset':float('nan')}
    with pytest.raises(FormatError):encode_lip(data)
