"""Independent readback of compatible replay timing, pictures and sound."""
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'backend/src'), str(ROOT / 'scripts')]
from verification_artifacts import verification_run
from ptb_worker.m05_playback import playback_bundle


@pytest.fixture
def scratch():
    with verification_run('m05-playback', 'codec', 'FFV1 VFR coloured frames at .2/.3/.5/.6s, PCM sine at .05 or .25s; 16kHz, 1s; optional silent input') as (_, path):
        yield path


def make_source(path, audio_start):
    import av
    with av.open(str(path), 'w', format='matroska') as out:
        video = out.add_stream('ffv1', rate=10)
        video.width = video.height = 64
        video.pix_fmt = 'bgr0'
        video.time_base = video.codec_context.time_base = Fraction(1, 1000)
        if audio_start is not None:
            sound = out.add_stream('pcm_s16le', rate=16000)
            sound.layout = 'mono'
            values = (8000 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype('<i2')[None, :]
            frame = av.AudioFrame.from_ndarray(values, format='s16', layout='mono')
            frame.sample_rate = 16000
            frame.time_base, frame.pts = Fraction(1, 16000), round(audio_start * 16000)
            for packet in sound.encode(frame):
                out.mux(packet)
            for packet in sound.encode():
                out.mux(packet)
        for t in [200, 300, 500, 600]:
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, :, 1] = 180
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.time_base, frame.pts = Fraction(1, 1000), t
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)


@pytest.mark.parametrize('audio_start', [.05, .25, None])
def test_playback_preserves_vfr_frames_audio_delay_and_signal(scratch, audio_start):
    import av
    source = scratch / 'source.mkv'
    make_source(source, audio_start)
    info = playback_bundle(source, scratch)
    origin = min(.2, audio_start) if audio_start is not None else .2
    assert info['origin_s'] == pytest.approx(origin)
    assert info['video_frames'] == 4
    with av.open(str(scratch / info['name'])) as media:
        frames = list(media.decode(video=0))
        assert len(frames) == 4
        assert [float(f.pts * f.time_base) for f in frames] == pytest.approx([t - origin for t in [.2, .3, .5, .6]], abs=.001)
        assert np.mean(frames[0].to_ndarray(format='rgb24')[:, :, 1]) > 150
    with av.open(str(scratch / info['name'])) as media:
        assert bool(media.streams.audio) == (audio_start is not None)
        if audio_start is not None:
            frames = list(media.decode(audio=0))
            assert float(frames[0].pts * frames[0].time_base) == pytest.approx(audio_start - origin, abs=.003)
            assert sum(f.samples for f in frames) / 48000 == pytest.approx(1., abs=.003)
            assert max(float(np.max(np.abs(f.to_ndarray()))) for f in frames) > .1
