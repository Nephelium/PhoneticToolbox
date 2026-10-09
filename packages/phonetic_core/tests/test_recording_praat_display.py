"""M16-R4 independent Praat calls, analytic tones and bounded streaming."""
import numpy as np
import parselmouth
import pytest
from phonetic_core.recording.praat_display import PraatDisplaySpectrum, spectrum


@pytest.mark.parametrize('sr',[8000,16000,44100,48000,96000])
def test_praat_analysis_and_grayscale_against_direct_reference(sr):
    t=np.arange(sr)/sr
    x=np.column_stack((.3*np.sin(2*np.pi*1000*t),.2*np.sin(2*np.pi*3000*t))).astype('float32')
    original=x.copy()
    display=PraatDisplaySpectrum(len(x),sr,1,width=16)
    for i in range(0,len(x),137):display.add(x[i:i+137])
    out=display.result();powers=[]
    for a in display.starts:
        # Independent direct reference: extract from original, run upstream Praat.
        window=x[a:a+display.window_frames,1].astype('float64')
        ref=parselmouth.Sound(window,sampling_frequency=sr).to_spectrogram(
            window_length=.005,maximum_frequency=min(5000,sr/2),time_step=1,
            frequency_step=max(20,min(5000,sr/2)/250),window_shape=parselmouth.SpectralAnalysisWindowShape.GAUSSIAN)
        powers.append(ref.values[:,0])
    power=np.asarray(powers).T;db=10*np.log10(np.maximum(power,np.finfo(float).tiny))
    np.testing.assert_array_equal(out['rows'],db.T)
    db+=6*np.log2(np.maximum(ref.ys(),ref.dy/2)/1000)[:,None]
    expected=np.rint(255*np.clip((db.max()-db)/50,0,1)).astype('uint8').T
    np.testing.assert_array_equal(out['pixels'],expected)
    peaks=np.asarray(out['frequencies'])[np.argmax(out['rows'],axis=1)]
    assert np.all(abs(peaks-3000)<=ref.dy)
    assert out['display_revision']=='m16-praat-display/1'
    assert out['time_edges'][0]==0 and out['time_edges'][-1]==1
    np.testing.assert_array_equal(x,original)


def test_praat_streaming_silence_short_and_long_memory():
    rng=np.random.default_rng(164)
    x=rng.normal(0,.1,(48000*3,2)).astype('float32')
    whole=spectrum(x,48000,0);display=PraatDisplaySpectrum(len(x),48000,0,width=128)
    for i in range(0,len(x),733):display.add(x[i:i+733])
    assert whole==display.result()
    for n in [0,1,31,96000]:
        out=spectrum(np.zeros((n,1),'float32'),48000)
        assert all(v==255 for row in out['pixels'] for v in row)
        assert len(out['rows'])<=128
    long=PraatDisplaySpectrum(96000*3600,96000)
    assert long.windows.nbytes<=640*1024*4
    with pytest.raises(ValueError,match='完整'):long.result()
    with pytest.raises(ValueError,match='通道'):spectrum(np.zeros((50,1),'float32'),48000,1)
