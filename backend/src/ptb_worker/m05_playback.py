"""Bounded VP8/Opus playback derivative, preserving the source A/V clock."""
from fractions import Fraction
from pathlib import Path
import time
import wave


def playback_bundle(source, directory):
    import av
    import numpy as np
    from .m05_audio import extract_audio
    from .m05_video import sha256_file

    source, directory = Path(source), Path(directory)
    output = directory / 'playback.webm'
    started = time.monotonic()

    def check():
        if time.monotonic() - started > 600:
            raise ValueError('回放视频转换超时，请缩短视频后重试。')
        if output.exists() and output.stat().st_size > 128_000_000:
            raise ValueError('回放视频超过 128 MB 预算，请分段分析。')
        return False

    if not 0 < source.stat().st_size <= 128_000_000:
        raise ValueError('回放输入超过 128 MB 预算。')
    with av.open(str(source)) as media:
        if not media.streams.video:
            raise ValueError('原文件没有视频流。')
        stream = media.streams.video[0]
        first = next(media.decode(stream), None)
        if first is None or first.pts is None or first.time_base is None:
            raise ValueError('原视频缺少有效时间戳。')
        first_video = first.pts * first.time_base
        width, height = first.width, first.height
        angle = int(first.rotation) % 360
        if angle not in (0, 90, 180, 270) or width * height > 1920 * 1080:
            raise ValueError('视频方向或尺寸超出回放范围。')
        if angle in (90, 270):
            width, height = height, width
        rate = stream.average_rate or Fraction(30)
    audio, _ = extract_audio(source, directory, stop=check)
    origin = min(first_video, Fraction(str(audio['first_decoded_pts_s'])) if audio['present'] else first_video)
    times = []
    with output.open('xb') as handle, av.open(str(source)) as media, av.open(handle, 'w', format='webm') as out:
        video = out.add_stream('libvpx', rate=rate)
        video.width, video.height, video.pix_fmt = width, height, 'yuv420p'
        video.time_base = video.codec_context.time_base = Fraction(1, 90000)
        video.bit_rate = 2_000_000
        video.options = {'deadline': 'realtime', 'cpu-used': '8', 'lag-in-frames': '0'}
        sound = out.add_stream('libopus', rate=48000) if audio['present'] else None
        if sound:
            sound.layout = 'mono' if audio['channels'] == 1 else 'stereo'
            sound.bit_rate = 128000
        samples = 0
        pcm = wave.open(str(directory / 'audio_recording.wav'), 'rb') if sound else None

        def audio_until(limit):
            nonlocal samples
            if pcm is None:
                return
            audio_origin = Fraction(str(audio['first_decoded_pts_s'])) - origin
            while samples < audio['samples'] and audio_origin + Fraction(samples, audio['sample_rate']) <= limit:
                check()
                raw = pcm.readframes(1024)
                values = np.frombuffer(raw, dtype='<i2').reshape(-1, audio['channels']).T.astype(np.float32) / 32768.
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(values), format='fltp', layout=sound.layout.name)
                frame.sample_rate = audio['sample_rate']
                frame.time_base = Fraction(1, audio['sample_rate'])
                frame.pts = round(audio_origin * audio['sample_rate']) + samples
                for packet in sound.encode(frame):
                    out.mux(packet)
                samples += values.shape[1]

        try:
            for frame in media.decode(video=0):
                check()
                if len(times) >= 300_000 or frame.pts is None or frame.time_base is None:
                    raise ValueError('回放帧数或时间戳无效。')
                t = frame.pts * frame.time_base - origin
                if times and t <= times[-1]:
                    raise ValueError('原视频时间戳不递增。')
                times.append(t)
                audio_until(t)
                image = frame.to_ndarray(format='rgb24')
                if angle:
                    image = np.ascontiguousarray(np.rot90(image, angle // 90))
                converted = av.VideoFrame.from_ndarray(image, format='rgb24')
                converted.pts, converted.time_base = round(t * 90000), Fraction(1, 90000)
                for packet in video.encode(converted):
                    out.mux(packet)
            for packet in video.encode():
                out.mux(packet)
            if sound:
                audio_until(float('inf'))
                for packet in sound.encode():
                    out.mux(packet)
        finally:
            if pcm:
                pcm.close()
    check()
    # WebM uses millisecond ticks. Verify frame count/timing and decoded audio.
    with av.open(str(output)) as media:
        decoded = [frame.pts * frame.time_base for frame in media.decode(video=0)]
    if len(decoded) != len(times) or any(abs(a - b) > Fraction(1, 1000) for a, b in zip(decoded, times)):
        raise ValueError('回放视频帧时间核对失败。')
    with av.open(str(output)) as media:
        if bool(media.streams.audio) != audio['present']:
            raise ValueError('回放音轨核对失败。')
        if audio['present']:
            decoded_audio = iter(media.decode(audio=0))
            first_audio = next(decoded_audio, None)
            if first_audio is None or abs(float(first_audio.pts * first_audio.time_base) - (audio['first_decoded_pts_s'] - float(origin))) > .003:
                raise ValueError('回放音频起点核对失败。')
            decoded_samples = first_audio.samples + sum(frame.samples for frame in decoded_audio)
            if abs(decoded_samples / 48000 - audio['samples'] / audio['sample_rate']) > .003:
                raise ValueError('回放音频时长核对失败。')
    return dict(name=output.name, size_bytes=output.stat().st_size, sha256=sha256_file(output),
                origin_s=float(origin), video_frames=len(times), audio_present=audio['present'])
