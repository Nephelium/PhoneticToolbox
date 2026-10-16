"""Bounded timestamp-driven animation rendering. Frames are visualization, not measurements."""
import math
import json
from pathlib import Path
from fractions import Fraction
from .m05_results import rows

PROFILES={'high':(1080,20),'standard':(720,24),'small':(540,28)}


def export_animation(frames_path,output,metadata,*,quality='standard',format='mp4',offset=0.,audio_path=None,stop=lambda:False,max_bytes=128_000_000,fit_face=False,full_mesh=False):
    import av
    import cv2
    import numpy as np
    from phonetic_core.lip.metrics import OUTER_LIP_LANDMARKS,INNER_LIP_LANDMARKS,FACE_OVAL
    if quality not in PROFILES or format not in ('mp4','gif') or not math.isfinite(offset) or abs(offset)>2:raise ValueError('m05_animation_config')
    output=Path(output)
    if output.exists():raise FileExistsError('m05_animation_exists')
    dimensions=metadata['coordinates']['resolutions']
    if len(dimensions)!=1:raise ValueError('m05_variable_dimensions_animation')
    source_width,source_height=dimensions[0]
    edge,crf=PROFILES[quality];scale=edge/max(source_width,source_height)
    width=max(2,int(source_width*scale)//2*2);height=max(2,int(source_height*scale)//2*2)
    rate=30
    bounds=None;edges=None
    if fit_face:
        for row in rows(frames_path):
            if stop():raise InterruptedError('m05_cancelled')
            if row.get('points'):
                points=np.asarray(row['points'],dtype=float)
                lo,hi=points.min(axis=0),points.max(axis=0)
                bounds=(lo,hi) if bounds is None else (np.minimum(bounds[0],lo),np.maximum(bounds[1],hi))
    if full_mesh:
        # The preview uses the same frozen adjacency. Loading the inference
        # package just for constants can exhaust the bounded encoder process.
        spec=json.loads((Path(__file__).resolve().parents[3]/'resources/m05/metric-spec.json').read_text('utf8'))
        edges=[(i,j) for i,neighbors in enumerate(spec['neighbors']) for j in neighbors if i<j]
    iterator=iter(rows(frames_path));previous=next(iterator,None)
    if previous is None:raise ValueError('m05_empty_animation')
    following=next(iterator,None);source_start=previous['time_s'];source_end=metadata['timing']['last_video_pts_s']-metadata['timing']['anchor_s']
    audio=None;audio_offset=0.;audio_samples=0
    if audio_path is not None:
        import wave
        audio=wave.open(str(audio_path),'rb')
        if audio.getsampwidth()!=2:audio.close();raise ValueError('m05_animation_audio_format')
        audio_offset=metadata['audio']['first_decoded_pts_s']-metadata['timing']['anchor_s']
    start=min(0.,source_start+offset,audio_offset if audio else 0.)
    end=max(source_end+offset,audio_offset+audio.getnframes()/audio.getframerate() if audio else source_end+offset)
    frames=0
    # PyAV writes directly to a bounded seekable file; no all-frame GIF list.
    class BoundedFile:
        def __init__(self):self.file=output.open('xb+')
        def write(self,data):
            if self.file.tell()+len(data)>max_bytes:raise ValueError('m05_animation_budget')
            return self.file.write(data)
        def seek(self,offset,whence=0):return self.file.seek(offset,whence)
        def tell(self):return self.file.tell()
        def flush(self):return self.file.flush()
    handle=BoundedFile()
    try:
        # MP4's default 1 kHz movie timescale rounds audio edit-list offsets.
        # Use the audio sample clock to preserve an explicitly requested offset.
        options={'movie_timescale':str(audio.getframerate() if audio else 90000)} if format=='mp4' else None
        with av.open(handle,'w',format='gif' if format=='gif' else 'mp4',options=options) as container:
            stream=container.add_stream('gif' if format=='gif' else 'libx264',rate=rate)
            stream.width=width;stream.height=height
            stream.pix_fmt='pal8' if format=='gif' else 'yuv420p'
            # A shared white-to-line palette preserves the thin antialiased
            # mesh. RGB8's coarse fixed cube loses faint interior edge pixels.
            palette=None
            if format=='gif':
                color=np.array([30,130,30] if full_mesh else [30,30,30])
                rgb=np.rint(255+np.arange(256)[:,None]/255*(color-255)).astype(np.uint8)
                palette=np.column_stack((np.full(256,255,np.uint8),rgb))
            if format=='mp4':stream.options={'crf':str(crf),'preset':'veryfast'}
            sound=None
            if audio and format=='mp4':
                sound=container.add_stream('aac',rate=audio.getframerate());sound.layout='mono' if audio.getnchannels()==1 else 'stereo'
            def audio_until(t):
                nonlocal audio_samples
                if not sound:return
                while audio_samples<audio.getnframes() and audio_offset+audio_samples/audio.getframerate()<=t:
                    if stop():raise InterruptedError('m05_cancelled')
                    raw=audio.readframes(1024)
                    samples=np.frombuffer(raw,dtype='<i2').reshape(-1,audio.getnchannels()).T.astype(np.float32)/32768.
                    af=av.AudioFrame.from_ndarray(np.ascontiguousarray(samples),format='fltp',layout=sound.layout.name)
                    af.sample_rate=audio.getframerate();af.time_base=Fraction(1,audio.getframerate())
                    af.pts=round((audio_offset-start)*audio.getframerate())+audio_samples
                    for packet in sound.encode(af):container.mux(packet)
                    audio_samples+=samples.shape[1]
            for index in range(max(1,math.ceil((end-start)*rate)+1)):
                if stop():raise InterruptedError('m05_cancelled')
                playback_time=start+index/rate;audio_until(playback_time);t=playback_time-offset
                while following is not None and following['time_s']<=t:
                    previous=following;following=next(iterator,None)
                points=None
                # Do not interpolate across detection loss. Display resampling is explicit metadata.
                if previous['detected'] and source_start<=t<=source_end:
                    points=np.asarray(previous['points'],dtype=np.float32)
                    if following is not None and following['detected'] and following['time_s']>previous['time_s']:
                        ratio=min(1,max(0,(t-previous['time_s'])/(following['time_s']-previous['time_s'])))
                        points=points+(np.asarray(following['points'],dtype=np.float32)-points)*ratio
                image=np.full((height,width,3),255,np.uint8)
                if points is not None:
                    if bounds is not None:
                        lo,hi=bounds;size=np.maximum(hi-lo,1e-6)
                        points=(points-(lo+hi)/2)*min(width*.94/size[0],height*.94/size[1])+np.array([width/2,height/2])
                    else:points=points*scale
                    points=np.rint(points).astype(np.int32)
                    if edges is not None:
                        for a,b in edges:
                            if a<len(points) and b<len(points):cv2.line(image,tuple(points[a]),tuple(points[b]),(30,130,30),1,cv2.LINE_AA)
                    else:
                        for contour in (OUTER_LIP_LANDMARKS,INNER_LIP_LANDMARKS,FACE_OVAL):
                            cv2.polylines(image,[points[contour]],True,(30,30,30),max(1,edge//540),cv2.LINE_AA)
                if format=='gif':
                    indices=np.rint((255-image[:,:,0].astype(np.float32))*255/225).astype(np.uint8)
                    frame=av.VideoFrame.from_ndarray((indices,palette),format='pal8')
                else:frame=av.VideoFrame.from_ndarray(image,format='bgr24')
                frame.pts=index;frame.time_base=Fraction(1,rate)
                for packet in stream.encode(frame):container.mux(packet)
                frames+=1
            for packet in stream.encode():container.mux(packet)
            if sound:
                audio_until(end+1)
                for packet in sound.encode():container.mux(packet)
    finally:
        handle.file.close()
        if audio:audio.close()
    return dict(format=format,quality=quality,width=width,height=height,fps=rate,frames=frames,
                source_start_s=source_start,output_origin_s=start,lip_manual_offset=offset,visualization_resampled=True,audio_included=audio is not None and format=='mp4',face_fitted=fit_face,full_mesh=full_mesh,
                warning='Animation time is visualization only; original per-frame timestamps remain authoritative')
