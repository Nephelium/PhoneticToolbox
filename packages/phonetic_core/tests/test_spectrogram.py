import numpy as np
import pytest
from phonetic_core.models.audio import AudioInput
from phonetic_core.spectrogram import spectrogram_preview


def test_real_praat_channels_frequency_peak_and_time_grid_are_preserved():
    fs=16000;t=(np.arange(fs)+.5)/fs
    audio=AudioInput(np.column_stack((np.sin(2*np.pi*1000*t),np.sin(2*np.pi*2500*t))),fs)
    before=audio.samples.copy()
    for channel,hz in ((0,1000),(1,2500)):
        result=spectrogram_preview(audio,channel=channel,start=.1,end=.9,width=300)
        power=result.pop('_power');frequencies=result.pop('_frequencies')
        assert abs(frequencies[np.argmax(power.mean(axis=1))]-hz)<100
        assert result['width']<=302 and result['height']<=252
        assert result['start']==.1 and result['end']==.9
        assert .1<=result['x1']<.12
        assert result['backend']=='praat' and result['parselmouth_version']=='0.4.7'
    np.testing.assert_array_equal(audio.samples,before)


def test_spectrogram_silence_short_audio_and_invalid_viewports():
    audio=AudioInput(np.zeros(16000),16000)
    result=spectrogram_preview(audio,channel=0,start=0,end=1,width=200)
    assert set(result['pixels'])=={255}
    for start,end,channel in ((0,0,0),(-1,1,0),(0,2,0),(0,1,1),(0,.001,0)):
        with pytest.raises(ValueError):spectrogram_preview(audio,channel=channel,start=start,end=end,width=200)
    with pytest.raises(ValueError):
        spectrogram_preview(AudioInput(np.full(16000,1e200),16000),channel=0,start=0,end=1,width=200)
