"""Explicit device identity, supported-format checks and bounded output routing."""
import hashlib
import json
import threading
from queue import Queue,Empty,Full
import numpy as np
from .storage import iter_audio,read_range


def devices(backend=None):
    if backend is None:
        import sounddevice as backend
    hosts=backend.query_hostapis();result=[]
    for i,d in enumerate(backend.query_devices()):
        item={'index':i,'name':d['name'],'hostapi':hosts[d['hostapi']]['name'],'inputs':int(d['max_input_channels']),'outputs':int(d['max_output_channels']),'default_rate':int(d['default_samplerate'])}
        item['id']=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()[:24];result.append(item)
    return result


def validate_config(config,backend=None):
    if backend is None:
        import sounddevice as backend
    candidates=devices(backend);device=next((d for d in candidates if d['id']==config.get('device')),None)
    if not device or device['inputs']<1:raise ValueError('输入设备身份已变化或不存在，请刷新并重新选择')
    rate=int(config.get('sample_rate',48000));channels=int(config.get('channels',min(2,device['inputs'])));roles=config.get('roles',['microphone']*channels)
    if rate not in (16000,22050,32000,44100,48000,88200,96000) or not 1<=channels<=min(8,device['inputs']):raise ValueError('采样率或输入通道数不支持')
    if len(roles)!=channels or any(r not in ('microphone','egg','other') for r in roles):raise ValueError('每个物理输入必须显式设置通道角色')
    gain=float(config.get('gain_db',0))
    if not np.isfinite(gain) or not -60<=gain<=24:raise ValueError('数字增益超出范围')
    backend.check_input_settings(device=device['index'],channels=channels,dtype='float32',samplerate=rate)
    return {'device':device['id'],'device_index':device['index'],'device_name':device['name'],'hostapi':device['hostapi'],'sample_rate':rate,'channels':channels,'roles':roles,'gain_db':gain,'sample_format':'float32','system_processing':'unknown','raw_semantics':'application-received PCM; no software gain/AGC/normalization'}


class PlaybackStartError(RuntimeError):
    """A failed output start whose native stream still needs an owning retry."""
    def __init__(self,message,player):super().__init__(message);self.player=player


class Playback:
    def __init__(self,root,spans,config,device_id,channel=0,volume=.7,backend=None,reference_spans=None):
        if backend is None:
            import sounddevice as backend
        found=next((d for d in devices(backend) if d['id']==device_id and d['outputs']>0),None)
        if not found:raise ValueError('请选择有效的扬声器或耳机设备')
        if not 0<=channel<config['channels'] or not 0<=volume<=1:raise ValueError('试听通道或音量无效')
        backend.check_output_settings(device=found['index'],channels=1,dtype='float32',samplerate=config['sample_rate'])
        self.backend=backend;self.root=root;self.spans=spans;self.reference_spans=reference_spans;self.channel=channel;self.volume=volume;self.queue=Queue(16)
        self.done=threading.Event();self.finished=threading.Event();self.error='';self.position=0;self.stream=None;self.current=np.empty(0,np.float32);self.offset=0
        self.worker=threading.Thread(target=self._read,daemon=True,name='ptb-m16-play-reader');self.worker.start()
        try:
            # Preload before opening the stream, bounded to one block wait.
            try:self.current=self.queue.get(timeout=3)
            except Empty:raise ValueError(self.error or '试听数据读取超时')
            self.stream=backend.OutputStream(device=found['index'],samplerate=config['sample_rate'],channels=1,dtype='float32',blocksize=1024,callback=self._callback,finished_callback=self.finished.set)
            self.stream.start()
        except BaseException as exc:
            try:self.stop()
            except Exception as cleanup:raise PlaybackStartError('试听启动失败且输出流尚未释放，请重试停止：'+str(cleanup),self) from exc
            raise

    def _read(self):
        try:
            offset=0
            for block in iter_audio(self.root,self.spans,8192):
                if self.reference_spans is not None:block=read_range(self.root,self.reference_spans,offset,offset+len(block))-block
                offset+=len(block)
                while not self.done.is_set():
                    try:self.queue.put((block[:,self.channel]*self.volume).copy(),timeout=.1);break
                    except Full:pass
                if self.done.is_set():break
        except BaseException as exc:self.error=str(exc)
        finally:self.finished_read=True

    def _callback(self,out,frames,clock,status):
        out.fill(0)
        if status:self.error='播放设备发生欠载或中断：'+str(status);raise self.backend.CallbackAbort
        pos=0
        while pos<frames:
            if self.offset>=len(self.current):
                try:self.current=self.queue.get_nowait();self.offset=0
                except Empty:
                    if getattr(self,'finished_read',False):raise self.backend.CallbackStop
                    self.error='试听缓冲不足，已停止';raise self.backend.CallbackAbort
            count=min(frames-pos,len(self.current)-self.offset);out[pos:pos+count,0]=self.current[self.offset:self.offset+count];pos+=count;self.offset+=count;self.position+=count

    def stop(self):
        self.done.set()
        failure=None
        if self.stream:
            stream=self.stream
            try:stream.abort()
            except Exception as exc:self.error='试听中止异常：'+str(exc)
            finally:
                try:stream.close();self.stream=None
                except Exception as exc:failure=exc;self.error='输出流关闭失败：'+str(exc)
        if hasattr(self,'worker'):self.worker.join(2)
        if failure is not None:raise RuntimeError('输出流仍未释放，请重试停止') from failure
        if hasattr(self,'worker') and self.worker.is_alive():raise RuntimeError('试听读取线程尚未结束，请重试停止')
        self.finished.set()
