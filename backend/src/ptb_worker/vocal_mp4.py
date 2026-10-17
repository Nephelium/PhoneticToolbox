"""M10 MP4 encoding in the existing isolated PyAV runtime.

Invoked only by the desktop file adapter with its own bounded scratch files.
No Qt, audio devices, network, or browser-provided paths are used here.
"""
from fractions import Fraction
import json
import math
from pathlib import Path
import sys


def encode(source, target, config):
    import av
    import numpy as np

    rate = 48000
    samples = round(config['duration'] * rate)
    frames = math.ceil(samples * 30 / rate)
    clock = Fraction(1, rate)
    count = audio_count = 0
    with av.open(str(source)) as inp, target.open('xb') as handle, av.open(
        handle, 'w', format='mp4',
        options={'movflags': '+faststart', 'movie_timescale': str(rate)},
    ) as out:
        video = out.add_stream('libx264', rate=30)
        video.width, video.height = config['width'], config['height']
        video.pix_fmt = 'yuv420p'
        video.time_base = video.codec_context.time_base = clock
        video.options = {'crf': '18', 'preset': 'veryfast', 'bf': '0'}
        audio = out.add_stream('aac', rate=rate)
        audio.layout = 'mono'
        audio.bit_rate = 128000
        audio.time_base = clock
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=rate)

        def mux(packets):
            for packet in packets:
                if packet.pts is not None and packet.pts >= 0:
                    remaining = Fraction(samples, rate) / packet.time_base - packet.pts
                    packet.duration = min(packet.duration or 1600, max(1, round(remaining)))
                out.mux(packet)

        for packet in inp.demux():
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame):
                    frame.pts, frame.time_base = count * 1600, clock
                    mux(video.encode(frame))
                    count += 1
                elif isinstance(frame, av.AudioFrame):
                    for pcm in resampler.resample(frame):
                        n = min(pcm.samples, samples - audio_count)
                        if n <= 0:
                            continue
                        values = np.ascontiguousarray(pcm.to_ndarray()[:, :n])
                        af = av.AudioFrame.from_ndarray(values, format='fltp', layout='mono')
                        af.sample_rate, af.time_base, af.pts = rate, clock, audio_count
                        mux(audio.encode(af))
                        audio_count += n
        if count != frames or audio_count != samples:
            raise ValueError('MP4 输入音视频时长不完整')
        mux(video.encode())
        mux(audio.encode())

    with av.open(str(target)) as checked:
        vs, sound = checked.streams.video[0], checked.streams.audio[0]
        if vs.codec_context.name != 'h264' or sound.codec_context.name != 'aac':
            raise ValueError('MP4 编码格式核对失败')
        if abs(vs.duration * vs.time_base - samples * clock) > clock:
            raise ValueError('MP4 视频时长核对失败')
        if abs(sound.duration * sound.time_base - samples * clock) > clock:
            raise ValueError('MP4 音轨时长核对失败')
        times = [f.pts * f.time_base for f in checked.decode(video=0)]
        if len(times) != frames or any(abs(t - Fraction(i, 30)) > clock for i, t in enumerate(times)):
            raise ValueError('MP4 帧数或时间戳核对失败')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ptb_worker.source_runtime import prepare_native_import_paths
    prepare_native_import_paths()
    request = json.loads(Path(sys.argv[1]).read_text('utf-8'))
    result = Path(sys.argv[1]).with_name('response.json')
    try:
        encode(Path(request['source']), Path(request['target']), request['config'])
        value = {'ok': True}
    except Exception as error:
        value = {'ok': False, 'error': str(error)}
    result.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
