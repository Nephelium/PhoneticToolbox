"""Display spectrum: analytic tones, streaming equivalence, bounded storage."""
import numpy as np
import pytest
from phonetic_core.recording.signal import DisplaySpectrum, spectrum


@pytest.mark.parametrize('sr', [8000, 16000, 48000, 96000])
def test_display_full_window_frequency_and_chunk_edges(sr):
    t = np.arange(sr*3)/sr
    frequencies = np.where(t < 1, 500, np.where(t < 2, 1500, 3000))
    x = (.3*np.sin(2*np.pi*frequencies*t)).astype('float32')[:,None]
    whole = spectrum(x, sr)
    stream = DisplaySpectrum(len(x),sr,width=128)
    for i in range(0,len(x),733):
        stream.add(x[i:i+733])
    assert stream.result() == whole
    assert stream.windows.nbytes <= 640*1024*4
    assert whole['time_edges'][0] == 0 and whole['time_edges'][-1] == 3
    assert whole['max_frequency'] == min(5000,sr/2)
    assert max(whole['frequencies']) <= min(5000,sr/2)
    for second,hz in [(.5,500),(1.5,1500),(2.5,3000)]:
        index = np.argmin(np.abs(np.asarray(whole['times'])-second))
        peak = whole['frequencies'][np.argmax(whole['rows'][index])]
        assert abs(peak-hz) <= sr/1024


def test_display_short_empty_and_huge_window_budget():
    assert spectrum(np.zeros((0,2),dtype='float32'),48000)['rows'] == []
    spec = spectrum(np.ones((31,2),dtype='float32'),48000,1)
    assert len(spec['rows']) == 1 and spec['time_edges'][-1] == 31/48000
    long = DisplaySpectrum(48000*3600,48000)
    assert long.windows.shape == (640,1024)
    with pytest.raises(ValueError,match='完整'):
        long.result()
    with pytest.raises(ValueError,match='通道'):
        spectrum(np.zeros((50,1),dtype='float32'),48000,1)
