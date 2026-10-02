"""Local recording conversion: every decoded frame keeps its presentation time."""
from fractions import Fraction
import json
from pathlib import Path
import sys
import time


def recording_bundle(source, directory, *, max_bytes=512_000_000):
    import av
    import numpy as np
    import wave
    from .m05_audio import extract_audio
    from .m05_video import sha256_file
    source, directory = Path(source), Path(directory)
    if source.stat().st_size > 128_000_000:
        raise ValueError('录制超过 128 MB 保存预算。')
    started = time.monotonic()
    def check():
        if time.monotonic() - started > 600:
            raise ValueError('MP4 转换超过 10 分钟期限，内存原录制仍保留。')
        if sum(p.stat().st_size for p in directory.iterdir() if p.is_file()) > max_bytes:
            raise ValueError('MP4/WAV 输出超过保存预算。')
        return False
    with av.open(str(source)) as inp:
        if not inp.streams.video:
            raise ValueError('录制没有视频流。')
        video = inp.streams.video[0]
        first = next(inp.decode(video), None)
        if first is None or first.pts is None:
            raise ValueError('录制没有有效视频时间戳。')
        first_video = first.pts * first.time_base
        width, height = first.width, first.height
        if width * height > 1920 * 1080 or width % 2 or height % 2:
            raise ValueError('MP4 保存支持偶数尺寸且不超过 1080p 的录制。')
        nominal = video.average_rate or Fraction(30)
    audio, audio_names = extract_audio(source, directory, stop=check,clock_policy='decoded_samples')
    origin = min(first_video, Fraction(str(audio['first_decoded_pts_s'])) if audio['present'] else first_video)
    output = directory / 'raw_recording.mp4'
    count = 0
    video_times = []
    with output.open('xb') as handle, av.open(str(source)) as inp, av.open(handle, 'w', format='mp4', options={'movflags': '+faststart', 'movie_timescale': str(audio['sample_rate'] if audio['present'] else 90000)}) as out:
        vs = inp.streams.video[0]
        dest = out.add_stream('libx264', rate=nominal)
        dest.width, dest.height, dest.pix_fmt = width, height, 'yuv420p'
        dest.time_base = dest.codec_context.time_base = Fraction(1, 90000)
        dest.options = {'crf': '18', 'preset': 'veryfast', 'bf': '0'}
        sound = None
        sound_samples = 0
        if audio['present']:
            sound = out.add_stream('aac', rate=audio['sample_rate'])
            sound.layout = 'mono' if audio['channels'] == 1 else 'stereo'
            sound.bit_rate = 192000
        pcm = wave.open(str(directory/'audio_recording.wav'), 'rb') if sound else None
        def audio_until(limit):
            nonlocal sound_samples
            if pcm is None:return
            sound_origin = Fraction(str(audio['first_decoded_pts_s'])) - origin
            while sound_samples < audio['samples'] and sound_origin + Fraction(sound_samples, audio['sample_rate']) <= limit:
                check()
                raw=pcm.readframes(1024)
                values=np.frombuffer(raw,dtype='<i2').reshape(-1,audio['channels']).T.astype(np.float32)/32768.
                af=av.AudioFrame.from_ndarray(np.ascontiguousarray(values),format='fltp',layout=sound.layout.name)
                af.sample_rate=audio['sample_rate'];af.time_base=Fraction(1,audio['sample_rate'])
                af.pts=round(sound_origin*audio['sample_rate'])+sound_samples
                for packet in sound.encode(af):out.mux(packet)
                sound_samples+=values.shape[1]
        try:
            for frame in inp.decode(vs):
                check()
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('录制缺少解码时间戳。')
                t = frame.pts * frame.time_base - origin
                if video_times and t <= video_times[-1]:
                    raise ValueError('录制视频时间戳不递增。')
                video_times.append(t)
                audio_until(t)
                frame.pts, frame.time_base = round(t * 90000), Fraction(1, 90000)
                for encoded in dest.encode(frame):
                    out.mux(encoded)
                count += 1
            for packet in dest.encode():
                out.mux(packet)
            if sound:
                audio_until(float('inf'))
                for packet in sound.encode():out.mux(packet)
        finally:
            if pcm:pcm.close()
    # Verify actual output stream/frame count and PTS before declaring a save.
    with av.open(str(output)) as checked:
        times = [f.pts * f.time_base for f in checked.decode(video=0)]
    if len(times) != count or any(abs(a-b) > Fraction(1, 90000) for a, b in zip(times, video_times)):
        raise ValueError('MP4 转换后帧数或时间戳核对失败。')
    with av.open(str(output)) as checked:
        if bool(checked.streams.audio) != audio['present']:
            raise ValueError('MP4 转换后音频流核对失败。')
        if audio['present']:
            count_audio=sum(f.samples for f in checked.decode(audio=0))
            if not audio['samples']<=count_audio<=audio['samples']+2048:
                raise ValueError('MP4 转换后音频采样数核对失败。')
    info = dict(schema='m05-recording-export/1', video_frames=count, source_first_video_pts_s=float(first_video),
                mp4_origin_s=float(origin), pts_preserved=True, audio=audio,
                observed_video_fps=(count-1)/float(video_times[-1]-video_times[0]) if count > 1 else None,
                original_sha256=sha256_file(source), video_codec='h264', audio_codec='aac' if audio['present'] else None,
                note='MP4 is a transcoded derivative; WAV is decoded PCM. No physical synchronization claim.')
    info['files'] = [dict(name=n, bytes=(directory/n).stat().st_size, sha256=sha256_file(directory/n)) for n in ['raw_recording.mp4', *audio_names]]
    capture_path = directory/'capture.m05-preview.json'
    if capture_path.is_file():
        capture = json.loads(capture_path.read_text('utf8'))
        frames = capture.get('frames', [])
        if frames:
            # Browser candidate times start at MediaRecorder.start's observed video
            # clock. Preserve that clock estimate and label the method separately.
            from .io.lip import encode_lip
            from .io.limits import LimitError
            anchor = audio['first_decoded_pts_s'] if audio['present'] else float(first_video)
            vectors = {k: [] for k in ('relative_times', 'area', 'outer_width', 'open', 'circularity')}
            for row in frames:
                vectors['relative_times'].append(row['time_s'] + float(first_video) - anchor)
                for key in ('area', 'outer_width', 'open', 'circularity'):
                    value = (row.get('metrics') or {}).get(key)
                    vectors[key].append(float('nan') if value is None else value)
            vectors['metadata'] = dict(time_alignment_mode='anchored_audio_start', audio_first_frame_time=0., lip_manual_offset=capture.get('lip_manual_offset', 0.))
            try:
                payload = encode_lip(vectors)
                (directory/'candidate.lip.json').write_bytes(payload)
                info['candidate_exchange'] = 'candidate.lip.json'
            except LimitError:
                info['candidate_exchange'] = None
                info['candidate_exchange_note'] = 'Candidate vectors exceed the shared safe-reader budget; full frames remain in capture.m05-preview.json.'
            info['candidate_backend'] = capture.get('backend')
            info['candidate_clock'] = 'observed video clock at MediaRecorder.start; frame-boundary uncertainty, not physical synchronization'
            info['candidate_frames'] = len(frames)
            info['candidate_observed_fps'] = (len(frames)-1)/(frames[-1]['time_s']-frames[0]['time_s']) if len(frames)>1 else None
            info['candidate_not_legacy'] = True
    (directory/'recording-export.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf8')
    return info


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root/'backend/src'), str(root/'packages/phonetic_core/src')]
    from ptb_worker.m05_media_export import recording_bundle
    request = Path(sys.argv[1])
    try:
        value = json.loads(request.read_text('utf8'))
        result = dict(ok=True, value=recording_bundle(value['source'], request.parent))
    except Exception as error:
        result = dict(ok=False, error=str(error))
    (request.parent/'export-response.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf8')
