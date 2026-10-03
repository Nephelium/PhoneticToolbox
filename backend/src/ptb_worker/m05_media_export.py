"""Local recording conversion: every decoded frame keeps its presentation time."""
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time


def _recording_bundle(source, directory, *, max_bytes=512_000_000, write_video=True, preserve_problematic_video=False):
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
    if audio['present']:
        # Audit decoded samples without amplifying, normalizing or replacing them.
        peak=0.;sum_squares=0.;sample_values=0
        with wave.open(str(directory/'audio_recording.wav'),'rb') as pcm:
            while raw:=pcm.readframes(65536):
                check();values=np.frombuffer(raw,dtype='<i2').astype(np.float64)/32768.
                peak=max(peak,float(np.max(np.abs(values))));sum_squares+=float(np.dot(values,values));sample_values+=values.size
        db=lambda value:float(20*np.log10(max(1e-6,value)))
        audio['signal']=dict(peak_dbfs=db(peak),rms_dbfs=db((sum_squares/max(1,sample_values))**.5),low_signal=peak<.001,threshold_dbfs=-60,
                             note='Digital sample level only; low level does not establish absence of speech.')
    origin = min(first_video, Fraction(str(audio['first_decoded_pts_s'])) if audio['present'] else first_video)
    output = directory / 'raw_recording.mp4'
    count = 0
    video_times = []
    fallback=None
    if not write_video or preserve_problematic_video:
        with av.open(str(source)) as inp:
            extension='mp4' if 'mp4' in inp.format.name else ('webm' if inp.streams.video[0].codec_context.name in ('vp8','vp9','av1') else 'mkv')
            for frame in inp.decode(video=0):
                check()
                if frame.pts is None or frame.time_base is None:raise ValueError('录制缺少解码时间戳。')
                video_times.append(frame.pts*frame.time_base-origin)
        count=len(video_times)
        if any(b<=a for a,b in zip(video_times,video_times[1:])):
            fallback='raw_recording.'+extension
        if write_video and not fallback:video_times=[];count=0
    encode_video=write_video and not fallback

    if encode_video:
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
                observed_video_fps=(count-1)/float(video_times[-1]-video_times[0]) if count > 1 and video_times[-1]>video_times[0] else None,
                original_sha256=sha256_file(source), video_codec='h264', audio_codec='aac' if audio['present'] else None,
                note='MP4 is a transcoded derivative; WAV is decoded PCM. No physical synchronization claim.')
    video_name='raw_recording.mp4' if encode_video else fallback if write_video else None
    if write_video and fallback:
        import shutil
        with source.open('rb') as src,(directory/fallback).open('xb') as dest:shutil.copyfileobj(src,dest)
        if sha256_file(directory/fallback)!=sha256_file(source):raise ValueError('原始视频保存校验失败。')
        info['video_timing_warning']='原始视频存在重复或倒退时间戳，已原样保存容器，未强行改写帧时间。请使用检查偏移量核对。'
        info['video_codec']='original';info['audio_codec']='original'
    info['video_file']=video_name
    info['video_transcoded']=bool(encode_video)
    info['video_pts_monotonic']=fallback is None

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
                # R3 rows use the observed recorder-start origin, not first video
                # PTS. Adding first_video again double-counted the track delay.
                shift = 0. if capture.get('clock_mapping', {}).get('candidate_time_base') == 'recording_start_estimate' else float(first_video)
                vectors['relative_times'].append(row['time_s'] + shift - anchor)
                for key in ('area', 'outer_width', 'open', 'circularity'):
                    value = (row.get('metrics') or {}).get(key)
                    vectors[key].append(float('nan') if value is None else value)
            vectors['metadata'] = dict(time_alignment_mode='anchored_audio_start', audio_first_frame_time=0., lip_manual_offset=capture.get('lip_manual_offset', 0.),source_backend=capture.get('backend','unknown'),source_status='candidate')
            try:
                payload = encode_lip(vectors)
                (directory/'candidate.lip.json').write_bytes(payload)
                info['candidate_exchange'] = 'candidate.lip.json'
                usable=sum(math.isfinite(t) and math.isfinite(v) for t,v in zip(vectors['relative_times'],vectors['open']))
                if audio['present'] and usable>=2:
                    (directory/'audio_recording.lip.json').write_bytes(payload)
                    info['associated_exchange']='audio_recording.lip.json'
                elif audio['present']:
                    info['associated_exchange_note']='有效唇形不足两帧，未自动关联；媒体和完整候选记录已保留。'
            except LimitError:
                info['candidate_exchange'] = None
                info['candidate_exchange_note'] = 'Candidate vectors exceed the shared safe-reader budget; full frames remain in capture.m05-preview.json.'
            info['candidate_backend'] = capture.get('backend')
            info['candidate_clock'] = 'observed video clock at MediaRecorder.start; asynchronous origin uncertainty is unknown; not physical synchronization'
            info['candidate_frames'] = len(frames)
            info['candidate_observed_fps'] = (len(frames)-1)/(frames[-1]['time_s']-frames[0]['time_s']) if len(frames)>1 else None
            info['candidate_not_legacy'] = True
    # Include the directly usable companion and capture evidence in the receipt.
    names=([video_name] if video_name else [])+audio_names
    names.extend(n for n in ('capture.m05-preview.json','candidate.lip.json','audio_recording.lip.json') if (directory/n).is_file())
    info['files']=[dict(name=n,bytes=(directory/n).stat().st_size,sha256=sha256_file(directory/n)) for n in names]
    check()
    (directory/'recording-export.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf8')
    return info


def inspect_recording(source, directory):
    """Decode clocks and a min/max waveform. No inference or user output files."""
    import av
    import wave
    import numpy as np
    from .m05_audio import extract_audio
    source, directory = Path(source), Path(directory)
    if source.stat().st_size > 128_000_000:raise ValueError('录制超过检查预算。')
    started=time.monotonic()
    def stop():
        if time.monotonic()-started>120:raise ValueError('音频检查超时，录制仍保留。')
        return False
    with av.open(str(source)) as c:
        first=next(c.decode(video=0),None) if c.streams.video else None
        video_start=float(first.pts*first.time_base) if first is not None and first.pts is not None else None
    audio,_=extract_audio(source,directory,clock_policy='decoded_samples',stop=stop)
    if not audio['present']:raise ValueError('媒体没有音轨，无法显示音频波形。')
    times=[];values=[]
    with wave.open(str(directory/'audio_recording.wav'),'rb') as w:
        step=max(1,math.ceil(w.getnframes()/12000));pos=0
        while raw:=w.readframes(step):
            stop();data=np.frombuffer(raw,dtype='<i2').reshape(-1,w.getnchannels())[:,0]/32768.
            lo,hi=int(np.argmin(data)),int(np.argmax(data))
            for i in sorted((lo,hi)):times.append((pos+i)/w.getframerate());values.append(float(data[i]))
            pos+=len(data)
    return dict(audio=audio,first_video_pts_s=video_start,waveform=dict(times=times,values=values,duration=audio['samples']/audio['sample_rate'],sampleRate=audio['sample_rate'],channels=audio['channels']))


def recording_bundle(source, directory, *, max_bytes=512_000_000, metadata_path=None):
    """Only selected media are published; encoding scratch belongs to this call."""
    import shutil
    import tempfile
    from .m05_video import sha256_file
    source,directory=Path(source),Path(directory)
    capture_path=Path(metadata_path) if metadata_path else directory/'capture.m05-preview.json'
    capture=json.loads(capture_path.read_text('utf8')) if capture_path.is_file() else {}
    options=capture.get('save_options')
    if options is None:
        legacy_capture=directory/'capture.m05-preview.json'
        if capture_path.is_file() and capture_path.resolve()!=legacy_capture.resolve():
            with capture_path.open('rb') as src,legacy_capture.open('xb') as dest:shutil.copyfileobj(src,dest)
        return _recording_bundle(source,directory,max_bytes=max_bytes)
    if set(options)!={'video','animation'} or any(type(v) is not bool for v in options.values()):raise ValueError('录制保存选项不正确。')
    mode=capture.get('settings',{}).get('mode')
    if mode not in ('realtime','record_then_analyze'):raise ValueError('仅录制模式可以保存。')
    if mode=='record_then_analyze' and options!={'video':True,'animation':False}:raise ValueError('高帧率模式仅保存视频音频。')
    offset=capture.get('lip_manual_offset',0.)
    if type(offset) not in (float,int) or not math.isfinite(offset) or abs(offset)>2:raise ValueError('偏移量须在 ±2 秒内。')
    with tempfile.TemporaryDirectory(prefix='ptb-m05-export-') as scratch:
        stage=Path(scratch)
        (stage/'capture.m05-preview.json').write_text(json.dumps(capture,ensure_ascii=False,allow_nan=False),'utf8')
        info=_recording_bundle(source,stage,max_bytes=max_bytes,write_video=options['video'],preserve_problematic_video=True)
        names=['audio_recording.wav'] if info['audio']['present'] else []
        if not info['audio']['present']:raise ValueError('录制没有可解码音频，未完成保存。')
        if options['video']:names.append(info['video_file'])
        if mode=='realtime':
            names.extend(n for n in ('capture.m05-preview.json','audio_recording.lip.json','candidate.lip.json') if (stage/n).is_file())
            if options['animation']:
                frames=capture.get('frames',[])
                if not any(r.get('points') for r in frames):raise ValueError('没有可用面部特征点，请取消动画选项后保存音频与唇形记录。')
                from .m05_animation import export_animation
                anchor=info['audio']['first_decoded_pts_s']
                shift=0. if capture.get('clock_mapping',{}).get('candidate_time_base')=='recording_start_estimate' else info['source_first_video_pts_s']
                frame_path=stage/'animation-frames.jsonl'
                with frame_path.open('x',encoding='utf8') as f:
                    for r in frames:f.write(json.dumps({**r,'time_s':r['time_s']+shift-anchor},allow_nan=False)+'\n')
                dimensions=frames[0]['input_resolution']
                metadata=dict(coordinates=dict(resolutions=[dimensions]),audio=info['audio'],timing=dict(anchor_s=anchor,last_video_pts_s=frames[-1]['time_s']+shift))
                animation_start=time.monotonic()
                def animation_stop():
                    if time.monotonic()-animation_start>300:raise ValueError('动画导出超时，录制仍保留。')
                    return False
                info['animation']=export_animation(frame_path,stage/'face_animation.mp4',metadata,offset=offset,audio_path=stage/'audio_recording.wav',fit_face=True,full_mesh=True,stop=animation_stop)
                names.append('face_animation.mp4')
        info['save_options']=options
        if sum((stage/n).stat().st_size for n in names)>max_bytes:raise ValueError('所选录制文件超过 512 MB 保存预算。')
        info['files']=[dict(name=n,bytes=(stage/n).stat().st_size,sha256=sha256_file(stage/n)) for n in names]
        (stage/'recording-export.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),'utf8')
        for name in [*names,'recording-export.json']:
            with (directory/name).open('xb') as dest,(stage/name).open('rb') as src:shutil.copyfileobj(src,dest)
            if sha256_file(directory/name)!=sha256_file(stage/name):raise ValueError('保存后的文件校验失败。')
        return info


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root/'backend/src'), str(root/'packages/phonetic_core/src')]
    from ptb_worker.m05_media_export import recording_bundle,inspect_recording
    request = Path(sys.argv[1])
    try:
        value = json.loads(request.read_text('utf8'))
        if value.get('action')=='inspect':
            with __import__('tempfile').TemporaryDirectory(prefix='ptb-m05-inspect-') as scratch:
                output=inspect_recording(value['source'],scratch)
        else:output=recording_bundle(value['source'],value.get('destination',request.parent),metadata_path=request.parent/'capture.m05-preview.json')
        result = dict(ok=True, value=output)
    except Exception as error:
        result = dict(ok=False, error=str(error))
    (request.parent/'export-response.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf8')
