"""M01-B: exact decoder equality against the pinned Praat library, no legacy imports."""
import numpy as np
import parselmouth
import pytest
from scipy.io import wavfile
from phonetic_core.models.audio import AudioInput
from phonetic_core.acoustic.reaper_codec import reaper_pcm16


@pytest.mark.parametrize('rate', [16000, 22050, 44100])
@pytest.mark.parametrize('dtype', ['uint8', 'int16', 'int32', 'float32', 'float64'])
@pytest.mark.parametrize('channels', [1, 2])
def test_praat_array_constructor_matches_real_wav_decoder(rate, dtype, channels, tmp_path):
    t = np.arange(1600) / rate
    signal = .4 * np.sin(2*np.pi*120*t)
    if channels == 2: signal = np.column_stack([signal, .2*np.cos(2*np.pi*180*t)])
    if dtype == 'uint8': samples = (signal*128+128).astype(dtype)
    elif dtype == 'int16': samples = (signal*32768).astype(dtype)
    elif dtype == 'int32': samples = (signal*2147483648).astype(dtype)
    else: samples = signal.astype(dtype)
    path = tmp_path/'probe.wav'
    wavfile.write(path, rate, samples)
    original = samples.copy()
    decoded = parselmouth.Sound(str(path))
    audio = AudioInput(samples, rate)
    adapted = audio.praat_sound()
    np.testing.assert_array_equal(adapted.values, decoded.values)
    assert adapted.xmin == decoded.xmin and adapted.xmax == decoded.xmax
    assert adapted.dx == decoded.dx and adapted.x1 == decoded.x1
    reaper_pcm16(audio)
    np.testing.assert_array_equal(samples, original)
    assert not audio.samples.flags.writeable


@pytest.mark.parametrize('samples,rate', [(np.zeros((2,2,2)),16000), ([np.nan],16000),
    ([np.inf],16000), (np.zeros(2),0), (np.zeros(2),True), (np.array([1],dtype='int64'),16000)])
def test_malformed_decoded_audio_rejected(samples, rate):
    with pytest.raises(ValueError): AudioInput(samples, rate)


def test_known_pcm_quantization_bytes():
    for data, expected in [
        (np.array([0,128,255],dtype=np.uint8), [-32768,0,32512]),
        (np.array([-2147483648,0,2147418112],dtype=np.int32), [-32768,0,32767]),
        (np.array([-.5,0.,.25],dtype=np.float32), [-32000,0,16000]),
    ]:
        assert reaper_pcm16(AudioInput(data,16000)).tobytes() == np.array(expected,dtype=np.int16).tobytes()


@pytest.mark.parametrize('rate', [16000, 22050, 44100])
def test_result_metadata_uses_actual_input_rate(rate):
    from phonetic_core.models.acoustic import AcousticConfig
    from phonetic_core.services.acoustic import analyze_audio
    result = analyze_audio(AudioInput(np.zeros(0, dtype=np.int16), rate), AcousticConfig(use_reaper=False))
    assert result.sampling_rate == rate
