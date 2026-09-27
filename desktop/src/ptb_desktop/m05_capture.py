"""Native disk-first capture adapter. No devices open on import.

Camera timestamps are host receive times, not claims about sensor exposure.
ADC timestamps and callback host mapping are retained separately. Physical
devices remain unverified; queue overflow ends capture with an incomplete flag.
"""
import json
import math
from pathlib import Path
import queue
import shutil
import threading
import time
from fractions import Fraction

def audio_clock_observation(host_ns,adc,current,frames,rate):
    mapped=host_ns+round((adc-current)*1e9) if math.isfinite(adc) and math.isfinite(current) else None
    valid=adc>0 and current>0 and mapped is not None and mapped>=0
    # Some WDM-KS drivers report ADC time as zero. Retain that fact instead of
    # turning repeated negative mappings into repeated encoded PTS=0.
    return (mapped if valid else max(0,host_ns-round(frames/rate*1e9))),valid

class CaptureMux:
    def __init__(self,path,width,height,rate,audio_rate,*,max_bytes=1_000_000_000):
        import av
        self.av=av;self.path=Path(path);self.file=self.path.open('xb+');self.max_bytes=max_bytes
        self.container=av.open(self,'w',format='matroska')
        self.video=self.container.add_stream('ffv1',rate=round(rate));self.video.width=width;self.video.height=height;self.video.pix_fmt='bgr0'
        # FFV1's default codec time base is 1/requested_fps. Override it so
        # arbitrary received timestamps are not quantized to a nominal grid.
        # Matroska timestamps have 1 ms granularity; original ns stay in sidecar.
        self.video.time_base=Fraction(1,1000);self.video.codec_context.time_base=Fraction(1,1000)
        self.audio=self.container.add_stream('pcm_s16le',rate=audio_rate);self.audio.layout='mono'
        self.audio_rate=audio_rate;self.last_video=-1;self.last_audio=-1
    def write(self,raw):
        if self.file.tell()+len(raw)>self.max_bytes:raise ValueError('native_recording_disk_budget')
        return self.file.write(raw)
    def seek(self,*args):return self.file.seek(*args)
    def tell(self):return self.file.tell()
    def flush(self):return self.file.flush()
    def video_frame(self,array,ns):
        if ns<=self.last_video:raise ValueError('nonmonotonic_camera_receive_time')
        self.last_video=ns;frame=self.av.VideoFrame.from_ndarray(array,format='bgr24');frame.pts=ns;frame.time_base=Fraction(1,1_000_000_000)
        for packet in self.video.encode(frame):self.container.mux(packet)
    def audio_frame(self,samples,ns):
        pts=round(ns*self.audio_rate/1e9)
        if pts<=self.last_audio:raise ValueError('nonmonotonic_audio_adc_time')
        self.last_audio=pts;frame=self.av.AudioFrame.from_ndarray(samples.reshape(1,-1),format='s16',layout='mono')
        frame.pts=pts;frame.time_base=Fraction(1,self.audio_rate);frame.sample_rate=self.audio_rate
        for packet in self.audio.encode(frame):self.container.mux(packet)
    def close(self):
        try:
            for stream in (self.video,self.audio):
                for packet in stream.encode():self.container.mux(packet)
            self.container.close();self.file.flush()
            import os
            os.fsync(self.file.fileno())
        finally:self.file.close()

def record(directory,*,camera_index,microphone_index,requested_fps=60,duration=60,width=1280,height=720,audio_rate=48000):
    import cv2
    import sounddevice as sd
    if not 1<=requested_fps<=240 or not 0<duration<=1800 or width*height>1920*1080:raise ValueError('native_capture_limits')
    target=Path(directory).absolute()
    if target.exists():raise FileExistsError('Capture output must be a new directory')
    target.mkdir(parents=True)
    if shutil.disk_usage(target).free<1_500_000_000:raise ValueError('native_capture_disk_space')
    stop=threading.Event();pending=queue.Queue(maxsize=128);lock=threading.Lock();buffered=[0];failure=[]
    stats=dict(camera_received=0,video_encoded=0,audio_callbacks=0,audio_encoded_samples=0,queue_peak_bytes=0,queue_rejected=0,first_video_ns=None,last_video_ns=None,audio_adc_unavailable=0)
    origin=time.perf_counter_ns();audio_origin=[None];audio_count=[0];timing=(target/'capture-timing.jsonl').open('x',encoding='utf8')
    camera=None;audio=None;mux=None;reader=None
    def enqueue(kind,data,t,extra):
        with lock:
            if buffered[0]+data.nbytes>16_000_000 or pending.full():
                stats['queue_rejected']+=1;failure.append('bounded_queue_overflow');stop.set();return
            buffered[0]+=data.nbytes;stats['queue_peak_bytes']=max(stats['queue_peak_bytes'],buffered[0]);pending.put_nowait((kind,data,t,extra))
    def sound(data,frames,stamp,status):
        host=time.perf_counter_ns()-origin
        # PortAudio ADC/current share a clock; map current callback to the host.
        adc,adc_valid=audio_clock_observation(host,stamp.inputBufferAdcTime,stamp.currentTime,frames,audio_rate)
        if not adc_valid:stats['audio_adc_unavailable']+=1
        if audio_origin[0] is None:audio_origin[0]=adc
        expected=audio_origin[0]+round(audio_count[0]/audio_rate*1e9)
        stats['audio_callbacks']+=1;audio_count[0]+=frames
        if status:failure.append('audio_'+str(status));stop.set()
        enqueue('audio',data.copy(),expected,dict(adc_time=stamp.inputBufferAdcTime,current_time=stamp.currentTime,host_ns=host,observed_adc_ns=adc if adc_valid else None,adc_valid=adc_valid,anchor_kind='adc_to_host' if adc_valid else 'estimated_callback_receive_minus_block_duration',mapping_residual_ns=adc-expected,status=str(status)))
    try:
        camera=cv2.VideoCapture(camera_index,cv2.CAP_DSHOW)
        if not camera.isOpened():raise ValueError('camera_open_failed')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,width);camera.set(cv2.CAP_PROP_FRAME_HEIGHT,height);camera.set(cv2.CAP_PROP_FPS,requested_fps)
        actual_width=int(camera.get(cv2.CAP_PROP_FRAME_WIDTH));actual_height=int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width*actual_height>1920*1080 or min(actual_width,actual_height)<1:raise ValueError('camera_resolution_limit')
        mux=CaptureMux(target/'original.mkv',actual_width,actual_height,requested_fps,audio_rate)
        def video():
            while not stop.is_set():
                ok,frame=camera.read();stamp=time.perf_counter_ns()-origin
                if not ok:failure.append('camera_read_failed');stop.set();return
                if stats['first_video_ns'] is None:stats['first_video_ns']=stamp
                stats['last_video_ns']=stamp
                stats['camera_received']+=1;enqueue('video',frame,stamp,{})
        reader=threading.Thread(target=video,daemon=True)
        audio=sd.InputStream(device=microphone_index,channels=1,samplerate=audio_rate,dtype='int16',callback=sound)
        audio.start();reader.start()
        try:
            while not stop.is_set() and (time.perf_counter_ns()-origin)/1e9<duration:
                try:item=pending.get(timeout=.1)
                except queue.Empty:continue
                kind,data,stamp,extra=item
                with lock:buffered[0]-=data.nbytes
                if kind=='video':mux.video_frame(data,stamp);stats['video_encoded']+=1
                else:mux.audio_frame(data,stamp);stats['audio_encoded_samples']+=len(data)
                timing.write(json.dumps(dict(kind=kind,time_ns=stamp,**extra))+'\n')
        except KeyboardInterrupt:pass
    except Exception as error:failure.append(type(error).__name__+':'+str(error))
    finally:
        stop.set()
        for name,cleanup in [('audio_stop',lambda:audio.stop() if audio else None),('audio_close',lambda:audio.close() if audio else None),('camera_release',lambda:camera.release() if camera else None)]:
            try:cleanup()
            except Exception as error:failure.append(name+':'+str(error))
        if reader:
            reader.join(timeout=5)
            if reader.is_alive():failure.append('camera_thread_did_not_stop')
        try:
            if mux:
                while not pending.empty():
                    kind,data,stamp,extra=pending.get_nowait()
                    if kind=='video':mux.video_frame(data,stamp);stats['video_encoded']+=1
                    else:mux.audio_frame(data,stamp);stats['audio_encoded_samples']+=len(data)
                    timing.write(json.dumps(dict(kind=kind,time_ns=stamp,**extra))+'\n')
                mux.close()
        except Exception as error:failure.append('finalization_failed:'+str(error))
        timing.close()
        elapsed=(time.perf_counter_ns()-origin)/1e9
        span=(stats['last_video_ns']-stats['first_video_ns'])/1e9 if stats['camera_received']>1 else 0
        result=dict(schema='m05-native-capture/1',complete=not failure,failures=failure,stats=stats,requested_fps=requested_fps,observed_receive_fps=(stats['camera_received']-1)/span if span else None,encoded_fps=(stats['video_encoded']-1)/span if span and stats['video_encoded']>1 else None,inference_fps=None,display_fps=None,duration_s=elapsed,timestamp='camera host receive / audio callback-estimated anchor + sample clock' if stats['audio_adc_unavailable'] else 'camera host receive / audio ADC mapped to host',audio_clock_calibrated=False,audio_adc_available=stats['audio_adc_unavailable']==0,actual_sensor_drops=None,device_validation='pending',original='original.mkv')
        (target/'capture.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result
