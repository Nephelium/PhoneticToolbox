"""Bounded WebM container writer for WebCodecs VP8/Opus packets.

Format reference: https://www.webmproject.org/docs/container/ (not copied code).
No encoder processes, filesystem paths, or audio devices are exposed to JS.
"""
import base64
import math
import struct
import uuid
from pathlib import Path


def uint(n):
    return n.to_bytes(max(1,(n.bit_length()+7)//8),'big')


def element(tag,payload):
    n=len(payload);size=max(1,(n.bit_length()+7)//7)
    return uint(tag)+((1<<(size*7))|n).to_bytes(size,'big')+payload


def number(tag,n):return element(tag,uint(n))
def string(tag,s):return element(tag,s.encode('utf-8'))


def webm(width,height,fps,duration,packets,opus):
    if len(opus)<19 or opus[:8]!=b'OpusHead' or opus[9]!=1:raise ValueError('无效的 Opus 声道描述')
    delay=struct.unpack_from('<H',opus,10)[0]/48000
    header=element(0x1A45DFA3,number(0x4286,1)+number(0x42F7,1)+number(0x42F2,4)+number(0x42F3,8)+string(0x4282,'webm')+number(0x4287,4)+number(0x4285,2))
    info=element(0x1549A966,number(0x2AD7B1,1000000)+element(0x4489,struct.pack('>d',duration*1000))+string(0x4D80,'PhoneticToolbox M10')+string(0x5741,'Qt WebCodecs'))
    video=element(0xAE,number(0xD7,1)+number(0x73C5,1)+number(0x83,1)+number(0x9C,0)+string(0x86,'V_VP8')+number(0x23E383,round(1e9/fps))+element(0xE0,number(0xB0,width)+number(0xBA,height)))
    audio=element(0xAE,number(0xD7,2)+number(0x73C5,2)+number(0x83,2)+number(0x9C,0)+string(0x86,'A_OPUS')+element(0x63A2,opus)+number(0x56AA,round(delay*1e9))+number(0x56BB,80000000)+element(0xE1,element(0xB5,struct.pack('>d',48000))+number(0x9F,1)))
    tracks=element(0x1654AE6B,video+audio)
    ordered=sorted(packets,key=lambda p:(p['timestamp'],-p['track']))
    clusters=[];cue_data=[];blocks=[];base=None
    last_video=max(p['timestamp'] for p in packets if p['track']==1);previous_video=0
    def flush():
        if blocks:clusters.append(element(0x1F43B675,number(0xE7,base)+b''.join(blocks)));blocks.clear()
    for p in ordered:
        stamp=max(0,round(p['timestamp']/1000))
        if base is None:base=stamp
        if p['track']==1 and p['key'] and stamp>base:
            flush();base=stamp
        if p['track']==1 and p['key']:cue_data.append((stamp,len(clusters)))
        payload=bytes([0x80|p['track']])+struct.pack('>h',stamp-base)+bytes([0x80 if p['key'] else 0])+p['data']
        padding=(p['timestamp']+p['duration'])/1e6-delay-duration if p['track']==2 else 0
        if p['track']==1 and p['timestamp']==last_video:
            # 50 ms poses need a partial final 30 fps frame. Store its exact
            # millisecond end rather than extending DefaultDuration past audio.
            reference=b'' if p['key'] else element(0xFB,(previous_video-stamp).to_bytes(8,'big',signed=True))
            blocks.append(element(0xA0,element(0xA1,payload[:3]+b'\0'+payload[4:])+number(0x9B,max(1,round(duration*1000)-stamp))+reference))
        elif padding>0:
            # Signed DiscardPadding removes the final Opus frame's extra samples.
            blocks.append(element(0xA0,element(0xA1,payload[:3]+b'\0'+payload[4:])+element(0x75A2,round(padding*1e9).to_bytes(8,'big',signed=True))))
        else:blocks.append(element(0xA3,payload))
        if p['track']==1:previous_video=stamp
    flush()
    # Fixed-size positions keep SeekHead and Cues offsets independent of values.
    pos=lambda tag,n:element(tag,n.to_bytes(8,'big'))
    def seek(i,t,c):
        return element(0x114D9B74,b''.join(element(0x4DBB,element(0x53AB,uint(tag))+pos(0x53AC,value)) for tag,value in [(0x1549A966,i),(0x1654AE6B,t),(0x1C53BB6B,c)]))
    def cues(offsets):
        return element(0x1C53BB6B,b''.join(element(0xBB,number(0xB3,t)+element(0xB7,number(0xF7,1)+pos(0xF1,offsets[i]))) for t,i in cue_data))
    sh=seek(0,0,0);cs=cues([0]*len(clusters));start=len(sh)+len(info)+len(tracks)+len(cs);offsets=[]
    for cluster in clusters:offsets.append(start);start+=len(cluster)
    sh=seek(len(sh),len(sh)+len(info),len(sh)+len(info)+len(tracks));cs=cues(offsets)
    return header+element(0x18538067,sh+info+tracks+cs+b''.join(clusters))


class VideoSession:
    def __init__(self,target,config):
        width=config.get('width');height=config.get('height');duration=config.get('duration');fps=config.get('fps')
        if type(width) is not int or type(height) is not int or not 64<=width<=1920 or not 64<=height<=1080 or fps!=30:
            raise ValueError('不支持的视频尺寸或帧率')
        if type(duration) not in (int,float) or not math.isfinite(duration) or not .1<=duration<=12:raise ValueError('视频时长应为 0.1–12 秒')
        self.id=uuid.uuid4().hex;self.target=Path(target);self.width=width;self.height=height;self.fps=fps;self.duration=round(duration*48000)/48000
        self.packets=[];self.size=0;self.opus=None;self.closed=False

    def append(self,obj):
        if self.closed:raise ValueError('视频导出会话已结束')
        packets=obj.get('packets')
        if not isinstance(packets,list) or not 1<=len(packets)<=100:raise ValueError('无效的视频数据批次')
        added=[];size=0
        for p in packets:
            track=p.get('track');timestamp=p.get('timestamp');duration=p.get('duration',0)
            if track not in (1,2) or type(timestamp) is not int or not -50000<=timestamp<=12050000 or type(duration) is not int or not 0<=duration<=100000:raise ValueError('无效的编码时间戳')
            raw=base64.b64decode(p['data'],validate=True)
            if not raw or len(raw)>2_000_000:raise ValueError('编码帧超过容量')
            size+=len(raw);added.append(dict(track=track,timestamp=timestamp,duration=duration,key=bool(p.get('key')),data=raw))
        if self.size+size>64_000_000 or len(self.packets)+len(added)>1200:raise ValueError('视频超过导出容量')
        if obj.get('opus'):
            opus=base64.b64decode(obj['opus'],validate=True)
            if len(opus)>64 or not opus.startswith(b'OpusHead'):raise ValueError('无效的音频编码描述')
            self.opus=opus
        self.packets.extend(added);self.size+=size
        return {'received':len(self.packets)}

    def finish(self):
        if self.closed:raise ValueError('视频导出会话已结束')
        videos=sorted((p for p in self.packets if p['track']==1),key=lambda p:p['timestamp'])
        if len(videos)!=math.ceil(round(self.duration*48000)*self.fps/48000) or not videos[0]['key'] or any(abs(p['timestamp']-round(i*1e6/self.fps))>1 for i,p in enumerate(videos)):
            raise ValueError('视频帧不完整，未覆盖目标文件')
        audios=[p for p in self.packets if p['track']==2]
        if not audios or not self.opus or max(p['timestamp']+p['duration'] for p in audios)<self.duration*1e6-1000:raise ValueError('音频数据不完整，未覆盖目标文件')
        raw=webm(self.width,self.height,self.fps,self.duration,self.packets,self.opus)
        temporary=self.target.with_name('.'+self.target.name+'.'+self.id+'.tmp')
        try:
            with temporary.open('xb') as f:f.write(raw);f.flush()
            temporary.replace(self.target)
        finally:
            # Only the unique temporary file created by this session is cleaned.
            if temporary.exists():temporary.unlink()
        self.closed=True;self.packets.clear()
        return {'saved':True,'name':self.target.name,'bytes':len(raw),'duration':self.duration,'frames':len(videos)}

    def cancel(self):self.closed=True;self.packets.clear();return {'cancelled':True}
