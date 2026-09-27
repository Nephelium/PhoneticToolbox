"""Bounded M12 display conversion; original bytes and timestamps stay authoritative.

Uses existing SoundFile 0.13.1 and SciPy 1.16.3 runtime dependencies.
Only the first channel enters the compact preview, matching M12 spectrogram/RMS.
"""
import hashlib
import io
import math
import wave

from .file_provider import FileAccessError

MAX_SOURCE_BYTES = 2_000_000_000
MAX_PREVIEW_BYTES = 64_000_000
MAX_SAMPLE_VALUES = 32_000_000


def source_hash(stream):
    if stream.seek(0, 2) > MAX_SOURCE_BYTES:
        raise FileAccessError('标注长音频读取支持最大 2 GB WAV。')
    stream.seek(0)
    digest = hashlib.sha256()
    while block := stream.read(1_048_576):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def preview_audio(stream, max_bytes=MAX_PREVIEW_BYTES):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    size = stream.seek(0, 2)
    stream.seek(0)
    if size > MAX_SOURCE_BYTES:
        raise FileAccessError('标注长音频读取支持最大 2 GB WAV。')
    budget = min(max_bytes, MAX_PREVIEW_BYTES)
    try:
        with sf.SoundFile(stream, mode='r') as audio:
            frames, rate, channels = audio.frames, audio.samplerate, audio.channels
            if audio.format not in ('WAV', 'WAVEX', 'RF64') or frames <= 0 or not 8000 <= rate <= 96000 or not 1 <= channels <= 8:
                raise FileAccessError('标注工作台支持 8–96 kHz、最多 8 声道的非空 WAV。')
            supported = audio.subtype in ('PCM_U8', 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE')
            compact = size > budget or frames * channels > MAX_SAMPLE_VALUES or audio.format == 'RF64' or not supported
            duration = frames / rate
            if compact:
                max_frames = min(MAX_SAMPLE_VALUES, (budget - 44) // 2)
                rates = sorted({rate, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000}, reverse=True)
                target_rate = next((r for r in rates if r <= rate and (frames * r + rate - 1) // rate <= max_frames), None)
                if target_rate is None:
                    raise FileAccessError('录音过长，8 kHz 单声道预览仍超过 64 MB，请分段打开。')
                divisor = math.gcd(rate, target_rate)
                up, down = target_rate // divisor, rate // divisor
                # Integer-second chunks and down-aligned context share one output grid.
                # Context exceeds resample_poly's default FIR support on either side.
                chunk_frames = rate * 10
                context = down * math.ceil(max(rate * .05, 12 * down / up) / down)
                output = io.BytesIO()
                with wave.open(output, 'wb') as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(target_rate)
                    for start in range(0, frames, chunk_frames):
                        end = min(frames, start + chunk_frames)
                        left, right = max(0, start - context), min(frames, end + context)
                        audio.seek(left)
                        block = audio.read(right - left, dtype='float32', always_2d=True)
                        if len(block) != right - left or not np.isfinite(block).all():
                            raise FileAccessError('WAV 数据不完整或含非有限采样值。')
                        mono = block[:, 0]
                        if up != down:
                            mono = resample_poly(mono, up, down)
                        offset = (start - left) * up // down
                        count = ((end - start) * up + down - 1) // down
                        samples = mono[offset:offset + count]
                        if len(samples) != count or not np.isfinite(samples).all():
                            raise FileAccessError('WAV 重采样结果不完整或含非有限采样值。')
                        pcm = np.clip(np.rint(samples * 32768), -32768, 32767).astype('<i2')
                        writer.writeframesraw(pcm.tobytes())
                payload = output.getvalue()
                note = f'轻量预览：{target_rate / 1000:g} kHz / 16 位 / 第一声道；原文件 {rate / 1000:g} kHz / {channels} 声道。'
            else:
                target_rate = rate
                payload = None
                note = ''
        # SoundFile leaves a supplied Python stream open. Hash original bytes, not PCM.
        sha = source_hash(stream)
        if payload is None:
            payload = stream.read(budget + 1)
        if len(payload) > budget:
            raise FileAccessError('转换后的预览超过内存预算。')
        return payload, sha, duration, note
    except (sf.LibsndfileError, RuntimeError, wave.Error) as exc:
        raise FileAccessError('WAV 读取或轻量预览转换失败，请检查音频格式。') from exc
