"""M03-R7 RIFF decoding checked against explicit integer/float reference values."""
import struct
import numpy as np
import pytest
from scipy.io import wavfile
from ptb_worker.egg_recording import Recording
from ptb_worker.spectrogram_preview import PreviewError

@pytest.mark.parametrize('dtype',[np.int16,np.int32,np.float32,np.float64])
def test_bounded_reads_match_independent_samples(tmp_path,dtype):
    values=np.array([[-11,29],[10,-4],[37,39]],dtype=dtype)
    path=tmp_path/'public.wav';wavfile.write(path,8000,values)
    recording=Recording(path)
    try:
        assert recording.frames==3 and not recording.long
        np.testing.assert_array_equal(recording.read(1,3),values[1:3].astype(np.float32))
    finally:recording.close()
    assert recording.file.closed

def test_pcm24_and_rf64_decode_signed_samples(tmp_path):
    expected=np.array([[-8388608,8388607],[-1,0],[1234,-5678]],dtype=np.int32)
    data=b''.join((int(v)&0xffffff).to_bytes(3,'little') for v in expected.flat)
    fmt=struct.pack('<HHIIHH',1,2,8000,48000,6,24)
    ds=struct.pack('<QQQI',100,len(data),3,0)
    raw=b'RF64'+struct.pack('<I',0xffffffff)+b'WAVEds64'+struct.pack('<I',28)+ds+b'fmt '+struct.pack('<I',16)+fmt+b'data'+struct.pack('<I',0xffffffff)+data
    path=tmp_path/'public24.wav';path.write_bytes(raw)
    recording=Recording(path)
    try:np.testing.assert_array_equal(recording.read(0,3),expected.astype(np.float32))
    finally:recording.close()

def test_duration_budget_at_original_rate(tmp_path):
    # Sparse disk fixture tests real frame metadata, without a large RAM array.
    path=tmp_path/'too-long.wav';frames=8000*1800+1;size=frames*4
    fmt=struct.pack('<HHIIHH',1,2,8000,32000,4,16)
    with path.open('wb') as f:
        f.write(b'RIFF'+struct.pack('<I',size+36)+b'WAVEfmt '+struct.pack('<I',16)+fmt+b'data'+struct.pack('<I',size))
        f.truncate(44+size)
    with pytest.raises(PreviewError,match='egg_input_budget'):Recording(path)

def test_malformed_riff_rejected_before_scanning(tmp_path):
    path=tmp_path/'broken.wav';path.write_bytes(b'RIFF'+struct.pack('<I',50)+b'WAVEfmt '+struct.pack('<I',16)+b'bad')
    with pytest.raises(PreviewError):Recording(path)
