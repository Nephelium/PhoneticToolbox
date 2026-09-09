"""Loopback HTTP application hosted by the app-owned worker."""
import argparse
import bisect
import zlib
import io
import json
import os
from pathlib import Path
import secrets
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import webbrowser

import numpy as np
from scipy.io import wavfile
from phonetic_toolbox.core.vocal_tract.engine import Engine
from phonetic_toolbox.core.vocal_tract.monitor import analyze_output
from phonetic_toolbox.core.vocal_tract.trajectory import validate_pitch_curve
from phonetic_toolbox.core.vocal_tract.source import validate_source
from .profile import ProfileStore

from .audio_output import LiveAudio, output_devices
from .animation import prepare_animation, validate_frames

class Workbench:
    def __init__(self,playback_allowed=True,*,resource_dir,profile_dir):
        self.engine=Engine(resource_dir=resource_dir); self.profile=ProfileStore(profile_dir); self.live=LiveAudio(self.engine,playback_allowed=playback_allowed,config_path=Path(profile_dir)/"audio-settings.json");self.animation=None;self.animation_id=0;self.animation_playing=False
        self.token=secrets.token_urlsafe(24); self.revision=0; self.section=65
        self.pose_lock=threading.Lock(); self.last_request=time.monotonic()
        self.audio_control=threading.RLock()

    def state(self): return {'audio_locked':not self.live.playback_allowed,'active':self.live.active,'mode':self.live.mode,'audio_error':self.live.error,'audio':dict(self.live.stats),'revision':self.revision,'device':self.live.device_key,'volume':self.live.volume,'gain_db':self.live.gain_db}

def create_server(port=0,*,playback_allowed=True,resource_dir,profile_dir,web_dir):
    web_dir=Path(web_dir).resolve()
    app=Workbench(playback_allowed,resource_dir=resource_dir,profile_dir=profile_dir)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(web_dir),**kwargs)
        def log_message(self,*args): pass
        def end_headers(self):
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            super().end_headers()
        def data(self,code,obj):
            data=json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        def valid_origin(self):
            origin=self.headers.get('Origin')
            return not origin or origin==f'http://127.0.0.1:{self.server.server_port}'
        def do_GET(self):
            if self.headers.get('Host')!=f'127.0.0.1:{self.server.server_port}' or not self.valid_origin():
                return self.data(403,{'error':'仅允许本机同源页面访问'})
            app.last_request=time.monotonic()
            path=urlparse(self.path).path
            if path=='/api/meta': return self.data(200,{**app.engine.metadata(),'token':app.token,'pid':os.getpid(),'api_version':4,'audio_locked':not app.live.playback_allowed,'output_devices':output_devices() if app.live.playback_allowed else [],'audio_settings':{'device':app.live.device_key,'volume':app.live.volume,'gain_db':app.live.gain_db}})
            if path=='/api/keyframes':
                try:return self.data(200,app.profile.load_sequence(app.engine))
                except (ValueError,KeyError,TypeError):return self.data(500,{'error':'关键帧配置损坏，原文件已保留'})
            if path=='/api/status': return self.data(200,app.state())
            if path.startswith('/api/'): return self.data(404,{'error':'Unknown endpoint'})
            # Only static assets in web/, no prototype sources or vendor binaries.
            resolved=Path(self.translate_path(self.path)).resolve()
            if not resolved.is_relative_to((web_dir).resolve()): return self.data(403,{'error':'Invalid path'})
            return super().do_GET()
        def do_POST(self):
            try:
                if not self.valid_origin() or self.headers.get('Host')!=f'127.0.0.1:{self.server.server_port}' or self.headers.get('X-Session')!=app.token:
                    return self.data(403,{'error':'会话已失效，请刷新页面'})
                n=int(self.headers.get('Content-Length','0'))
                if n<0 or n>16384: return self.data(413,{'error':'Request too large'})
                obj=json.loads(self.rfile.read(n) or b'{}')
                if not isinstance(obj,dict): raise ValueError('Expected object')
                app.last_request=time.monotonic(); app.live.heartbeat=time.monotonic()
                path=urlparse(self.path).path
                if not app.live.playback_allowed and (path in ('/api/preview','/api/audio/test','/api/audio/settings','/api/animation/play') or (path=='/api/live' and obj.get('active'))):
                    return self.data(403,{'error':'静音建模已锁定，禁止音频输出'})
                if path=='/api/keyframes':
                    frames=validate_frames(app.engine,obj['frames'],for_storage=True)
                    app.profile.save_frames(frames,validate_pitch_curve(obj.get('pitch_curve',[])))
                    return self.data(200,{'saved':len(frames)})
                if path=='/api/audio/monitor':
                    seconds=float(obj.get('seconds',3));window=float(obj.get('window_ms',20));hop=float(obj.get('hop_ms',10))
                    if not np.isfinite([seconds,window,hop]).all() or not 1<=seconds<=12:raise ValueError('无效的分析范围')
                    samples,end,generation=app.live.output_history.snapshot(seconds)
                    return self.data(200,analyze_output(samples,app.engine.sr,seconds=seconds,window_ms=window,hop_ms=hop,end=end,generation=generation))
                if path=='/api/heartbeat': return self.data(200,app.state())
                if path=='/api/audio/settings':
                    with app.audio_control:app.live.configure(obj.get('device'),obj.get('volume'),obj.get('gain_db'))
                    return self.data(200,app.state())
                if path=='/api/audio/test':
                    with app.audio_control:app.live.test_tone()
                    return self.data(200,app.state())
                if path=='/api/pose':
                    p=app.engine.validated(obj['params']); rev=obj['revision']; section=obj.get('section',65)
                    if type(rev) is not int or rev<0: raise ValueError('Invalid revision')
                    f0=float(obj.get('f0',125)); app.engine.glottis(f0)
                    source=validate_source(obj.get('source')) if 'source' in obj else None
                    with app.pose_lock:
                        if rev<=app.revision: return self.data(409,{'error':'过期的构形请求','revision':app.revision})
                        width=float(obj.get('lip_width',1));notice=None
                        if obj.get('keep_vowel',False):p,notice=app.engine.constrain_nasal_opening(p,width)
                        snapshot=app.engine.snapshot(p,int(np.clip(section,0,128)),width)
                        snapshot['nasal_constraint']=notice
                        snapshot['source']=source
                        app.revision=rev;app.live.params=p;app.live.f0=f0;app.live.lip_width=width;app.section=snapshot['section']
                        app.live.source=source
                        if app.live.active and app.live.mode=='live':app.live.tube=app.engine.prepare_tube(p,width)
                    return self.data(200,{**snapshot,'revision':rev})
                if path=='/api/live':
                    with app.audio_control:
                        if obj.get('active'): app.live.start()
                        else: app.live.stop()
                    return self.data(200,app.state())
                if path in ('/api/render','/api/preview'):
                    with app.audio_control:
                        app.live.stop()
                        p=app.engine.validated(obj.get('params',app.live.params))
                        duration=float(obj.get('duration',1.2));width=float(obj.get('lip_width',app.live.lip_width))
                        if not .3<=duration<=5:raise ValueError('试听时长应为 0.3–5 秒')
                        poses=[app.engine.presets[n] for n in ['a','i','u']] if obj.get('sequence')=='a-i-u' else [p.tolist(),obj.get('end') or p.tolist()]
                        frames=[{'params':q,'lip_width':width,'f0':app.live.f0,'duration':duration/len(poses)} for q in poses]
                        if app.live.source is not None:
                            for frame in frames:frame['source']=app.live.source
                        audio=prepare_animation(app.engine,frames,app.section,pictures_enabled=False)['audio']
                        if path=='/api/preview':
                            app.live.start(audio,mode='preview')
                            return self.data(200,app.state())
                    out=io.BytesIO(); wavfile.write(out,app.engine.sr,np.clip(audio*2.2,-1,1).astype(np.float32)); raw=out.getvalue()
                    self.send_response(200); self.send_header('Content-Type','audio/wav');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
                if path=='/api/animation/prepare':
                    with app.audio_control:
                        app.live.stop();app.animation_playing=False
                        frames=validate_frames(app.engine,obj['frames'])
                        # Clamp saved endpoints, then use one unmodified interpolated
                        # trajectory for audio and geometry throughout each movement.
                        if obj.get('keep_vowel',False):
                            for frame in frames:frame['params']=app.engine.constrain_nasal_opening(frame['params'],frame['lip_width'])[0].tolist()
                        prepared=prepare_animation(app.engine,frames,app.section,pitch_curve=validate_pitch_curve(obj.get('pitch_curve',[])))
                        app.animation=prepared;app.animation_id+=1
                    return self.data(200,{'id':app.animation_id,'duration':prepared['duration'],'count':len(prepared['pictures']),'frames':prepared['frames']})
                if path=='/api/animation/play':
                    with app.audio_control:
                        if app.animation is None or obj.get('id')!=app.animation_id:raise ValueError('动作序列已失效，请重新生成')
                        app.animation_playing=True;app.live.start(app.animation['audio'],mode='animation')
                    return self.data(200,app.state())
                if path=='/api/animation/frame':
                    if app.animation is None or obj.get('id')!=app.animation_id:raise ValueError('动作序列已失效')
                    seconds=min(app.animation['duration'],app.live.position())
                    index=max(0,bisect.bisect_right(app.animation['times'],seconds)-1)
                    state=json.loads(zlib.decompress(app.animation['pictures'][index]))
                    if app.animation_playing and not app.live.active:
                        app.animation_playing=False;app.live.params=app.engine.validated(state['params']);app.live.lip_width=state['lip_width'];app.live.f0=state['f0']
                        app.live.source=state.get('source')
                    return self.data(200,{'index':index,'time':seconds,'duration':app.animation['duration'],'active':app.live.active,'completed':app.live.completed,'audio_error':app.live.error,'state':state})
                if path=='/api/animation/stop':
                    with app.audio_control:app.live.stop()
                    return self.data(200,app.state())
                if path=='/api/shutdown':
                    # In-flight synthesis must never start playback after shutdown.
                    app.live.playback_allowed=False
                    app.live.stop(); self.data(200,{'closed':True}); threading.Thread(target=self.server.shutdown,daemon=True).start(); return
                return self.data(404,{'error':'Unknown endpoint'})
            except (ValueError,KeyError,TypeError) as exc: self.data(400,{'error':str(exc)})
            except (BrokenPipeError,ConnectionResetError): pass
            except Exception as exc:
                app.live.active=False; self.data(500,{'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    # server_close joins outstanding native computations before Engine.close.
    # The owner can still terminate this process after its bounded exit timeout.
    server.daemon_threads=False
    server.app=app
    return server
