"""One owned PortAudio input stream, bounded queue and durable raw chunks.

Reviewed source: VoiceVista egg_recorder audio/{recorder,wav_writer}.py.
Reimplemented for arbitrary channel roles, pre-gain raw PCM and v3 lifecycle.
"""
from __future__ import annotations
import collections
import copy
import shutil
import threading
import time
from queue import Queue, Full, Empty
import numpy as np
from phonetic_core.recording import meter, spectrum
from .storage import save_pcm, atomic_json, CHUNK_FRAMES, uid, now


class Capture:
    def __init__(self,root,config,task=None,*,probe=False,backend=None,queue_blocks=64,chunk_frames=CHUNK_FRAMES):
        self.root=root;self.config=copy.deepcopy(config);self.task=copy.deepcopy(task);self.probe=probe
        self.backend=backend;self.queue=Queue(maxsize=queue_blocks);self.chunk_frames=chunk_frames
        self.id=uid();self.received=0;self.written=0;self.error='';self.done=threading.Event();self.ready=threading.Event()
        self.spans=[];self.events=[];self.stream=None;self.thread=None;self.live_lock=threading.Lock()
        self.live=collections.deque();self.live_frames=0;self.latest={};self.started=now();self.last_callback=time.monotonic()
        self.raw_clips=np.zeros(config['channels'],dtype=np.int64);self.near_clips=np.zeros(config['channels'],dtype=np.int64)
        self.peaks=np.zeros(config['channels']);self.square_sums=np.zeros(config['channels']);self.start_clock=0.
        self.preview_gain_db=float(config.get('gain_db',0))

    def snapshot(self):
        return {'id':self.id,'started_at':self.started,'config':self.config,'task_snapshot':self.task,
                'spans':copy.deepcopy(self.spans),'frames':self.written,'received_frames':self.received,
                'events':copy.deepcopy(self.events),'error':self.error,'status':'recovered_partial' if self.error else 'raw_saved',
                'quality':{'raw_clip':self.raw_clips.tolist(),'near_full_scale':self.near_clips.tolist(),'peak':self.peaks.tolist(),
                           'rms':np.sqrt(self.square_sums/max(1,self.written)).tolist()}}

    def fail(self,message):
        if not self.error:self.error=message
        self.done.set()

    def submit(self,indata,status=''):
        if self.done.is_set():return
        if self.received and time.monotonic()-self.last_callback>3:
            self.fail('采样时间出现超过 3 秒的中断，已停止且保留此前片段');return
        self.last_callback=time.monotonic()
        if status:
            self.fail('输入流异常：'+str(status));return
        # The callback only owns a bounded copy and a nonblocking enqueue.
        block=np.asarray(indata,dtype=np.float32)
        if block.ndim!=2 or block.shape[1]!=self.config['channels'] or len(block)>8192:
            self.fail('采集块格式或大小变化');return
        at=self.received
        try:self.queue.put_nowait((at,block.copy()))
        except Full:self.fail('采集写盘队列已满，录音不完整');return
        self.received+=len(block)

    def start(self,*,open_stream=True):
        self.thread=threading.Thread(target=self._write,name='ptb-m16-writer',daemon=True);self.thread.start()
        if not self.ready.wait(5):raise RuntimeError('录音写盘线程未就绪')
        if self.error:raise RuntimeError(self.error)
        self.start_clock=time.monotonic()
        if open_stream:
            try:
                if self.backend is None:
                    import sounddevice
                    self.backend=sounddevice
                cfg=self.config
                self.stream=self.backend.InputStream(device=cfg['device_index'],samplerate=cfg['sample_rate'],channels=cfg['channels'],dtype='float32',blocksize=1024,
                    callback=lambda data,frames,clock,status:self.submit(data,status))
                if round(self.stream.samplerate)!=cfg['sample_rate']:raise ValueError('设备实际采样率与请求不符，未开始录音')
                self.stream.start()
            except BaseException as exc:
                self.fail('无法打开输入设备：'+str(exc));self.stop();raise

    def _write(self):
        pending=[];count=0
        try:
            if not self.probe:
                atomic_json(self.root/'recovery'/self.id/'capture.json',self.snapshot())
            self.ready.set()
            while not self.done.is_set() or not self.queue.empty():
                try:at,block=self.queue.get(timeout=.1)
                except Empty:continue
                if at!=self.written+count:raise ValueError('采样帧序号不连续，已停止')
                if not np.isfinite(block).all():raise ValueError('输入含非有限采样，已停止')
                metrics=meter(block,self.config.get('gain_db',0));self.raw_clips+=metrics['raw_clip'];self.near_clips+=metrics['near_full_scale']
                self.peaks=np.maximum(self.peaks,metrics['peak']);self.square_sums+=np.sum(block.astype(np.float64)**2,axis=0)
                if any(metrics['raw_clip']) and len(self.events)<1024:self.events.append({'frame':at,'type':'raw_clip','counts':metrics['raw_clip']})
                with self.live_lock:
                    self.latest=metrics;self.live.append(block);self.live_frames+=len(block)
                    while self.live_frames>self.config['sample_rate']*5 and len(self.live)>1:self.live_frames-=len(self.live.popleft())
                if self.probe:self.written+=len(block);continue
                pending.append(block);count+=len(block)
                if count>=self.chunk_frames:
                    data=np.concatenate(pending);offset=0
                    while len(data)-offset>=self.chunk_frames:
                        self._flush(data[offset:offset+self.chunk_frames]);offset+=self.chunk_frames
                    pending=[data[offset:].copy()] if offset<len(data) else [];count=len(data)-offset
            if count and not self.probe:self._flush(np.concatenate(pending))
        except BaseException as exc:self.fail('录音写入失败：'+str(exc))
        finally:
            if not self.probe:
                try:atomic_json(self.root/'recovery'/self.id/'capture.json',self.snapshot())
                except OSError as exc:self.fail('录音清单写入失败：'+str(exc))
            self.ready.set()

    def _flush(self,data):
        if shutil.disk_usage(self.root).free<max(64*1024*1024,data.nbytes*2):raise OSError('磁盘剩余不足 64 MiB，已安全停止')
        span=save_pcm(self.root,f'takes/{self.id}/raw/{len(self.spans):06d}.f32',data)
        self.spans.append(span);self.written+=len(data)
        atomic_json(self.root/'recovery'/self.id/'capture.json',self.snapshot())

    def preview(self,show_spectrum=False,channel=0,gain_preview_db=None):
        if self.stream is not None and not self.done.is_set():
            try:
                if not self.stream.active or time.monotonic()-self.last_callback>3:self.fail('设备流中断或长时间未提供采样，录音不连续')
            except Exception:self.fail('设备已断开')
        with self.live_lock:
            data=np.concatenate(list(self.live)) if self.live else np.empty((0,self.config['channels']),np.float32)
            latest=copy.deepcopy(self.latest)
        if gain_preview_db is not None:
            gain=float(gain_preview_db);latest=meter(data[-min(len(data),4096):],gain)
            if gain!=self.preview_gain_db:
                self.preview_gain_db=gain
                if len(self.events)<1024:self.events.append({'frame':self.received,'type':'preview_gain','gain_db':gain,'raw_unchanged':True})
        out=preview_array(data,self.config['sample_rate'],show_spectrum,channel)
        out.update({'start_frame':max(0,self.received-len(data)),'frames':self.received,'confirmed_frames':self.written,'meter':latest,'error':self.error,
                    'queue_blocks':self.queue.qsize(),'recording':not self.probe and not self.done.is_set(),'probing':self.probe and not self.done.is_set()})
        return out

    def stop(self):
        if self.stream is not None:
            stream=self.stream;stop_error=None
            try:stream.stop()
            except Exception as exc:stop_error=exc;self.fail('输入流停止异常：'+str(exc))
            finally:
                try:stream.close();self.stream=None
                except Exception as exc:
                    self.fail('输入流关闭失败：'+str(exc));raise RuntimeError('输入流仍未释放，请重试停止') from exc
        self.done.set()
        if self.thread:
            self.thread.join(10)
            if self.thread.is_alive():raise RuntimeError('写盘仍未完成，工程保持打开，请稍后重试')
        return self.snapshot()


def preview_array(data,sample_rate,show_spectrum=False,channel=0,width=900):
    width=max(32,min(int(width),1600));n=len(data);wave=[]
    if n:
        edges=np.linspace(0,n,min(n,width)+1,dtype=int)
        for ch in range(data.shape[1]):
            wave.append([[float(data[a:b,ch].min()),float(data[a:b,ch].max())] for a,b in zip(edges[:-1],edges[1:])])
    return {'wave':wave,'window_frames':n,'sample_rate':sample_rate,'spectrum':spectrum(data,sample_rate,channel) if show_spectrum else None}
