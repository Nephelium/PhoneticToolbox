"""Owned, finite-resource VTL worker. JSON lines, no network or audio device."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import base64
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor


def configure_cpu_affinity():
    """An operator-selected Linux compute CPU set, before native threads start."""
    if sys.platform!='linux':return
    configured=os.environ.get('PTB_WEB_COMPUTE_CPUS')
    if configured is None:return
    try:cpus={int(value) for value in configured.split(',')}
    except ValueError:raise RuntimeError('Invalid configured compute CPUs') from None
    if not cpus or not cpus.issubset(os.sched_getaffinity(0)):
        raise RuntimeError('Configured compute CPUs unavailable')
    os.sched_setaffinity(0,cpus)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resources',type=Path,required=True)
    args=parser.parse_args()
    configure_cpu_affinity()
    if sys.platform=='linux':
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(1_073_741_824,1_073_741_824))
        resource.setrlimit(resource.RLIMIT_CPU,(600,600));resource.setrlimit(resource.RLIMIT_NOFILE,(64,64));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_FSIZE,(128_000_000,128_000_000))
    # Native library chatter must never corrupt the protocol.
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',encoding='utf-8',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    from phonetic_core.vocal_tract.engine import Engine
    from phonetic_core.vocal_tract.session import Runtime
    from .vocal_web import BrowserAudio,MemoryProfile,validate_web_operation,WEB_SEQUENCE_SECONDS
    runtime=Runtime(Engine(resource_dir=args.resources),MemoryProfile(),None)
    runtime.engine.enable_web_sections();runtime.section=runtime.engine.section_count//2
    runtime.live=BrowserAudio(runtime.engine)
    video=None;video_scratch=None
    write_lock=threading.Lock()
    def dispatch(message):
        nonlocal video,video_scratch
        try:
            try:
                op=message['op'];body=message.get('body',{})
                if not isinstance(body,dict):raise ValueError('无效的声道请求')
                validate_web_operation(runtime,op,body)
                if op=='video/begin':
                    from .vocal_video import VideoSession
                    if video is not None and not video.closed:raise ValueError('已有视频正在导出')
                    if shutil.disk_usage(tempfile.gettempdir()).free<256_000_000:raise ValueError('视频导出临时空间不足')
                    if video_scratch:video_scratch.cleanup()
                    video_scratch=tempfile.TemporaryDirectory(prefix='video-',dir=os.environ['PTB_VOCAL_SCRATCH'])
                    video=VideoSession(Path(video_scratch.name)/'声道动作.mp4',body,scratch_dir=video_scratch.name);value={'id':video.id}
                elif op in ('video/chunks','video/finish','video/cancel'):
                    if video is None or body.get('id')!=video.id:raise ValueError('视频导出会话已失效')
                    if op=='video/chunks':value=video.append(body)
                    elif op=='video/cancel':
                        value=video.cancel();video_scratch.cleanup();video=None;video_scratch=None
                    else:
                        try:
                            value=video.finish()
                            if video.target.stat().st_size>96_000_000:raise ValueError('视频超过下载容量')
                            value['base64']=base64.b64encode(video.target.read_bytes()).decode()
                        finally:video_scratch.cleanup();video=None;video_scratch=None
                elif op=='audio/pull':value=runtime.live.pull(body)
                else:
                    if op in ('animation/frame','audio/monitor','animation/stop','deactivate','live') and 'position' in body:runtime.live.acknowledge(body)
                    if op in ('heartbeat','status') and body.get('completed') is True:
                        # No new PCM is produced after EOF; completing here is safe
                        # and lets lightweight heartbeats bypass a long render.
                        runtime.live.acknowledge(body)
                    value=runtime.invoke(op,body,generation=message.get('generation'))
                if op=='meta':value.update(continuous_audio=False,preview_duration=.5,max_animation_duration=WEB_SEQUENCE_SECONDS)
                if op in ('status','heartbeat','preview','live','audio/test','animation/play'):
                    value['playback_generation']=runtime.live.output_history.generation
                result={'ok':True,'value':value}
            except (ValueError,TypeError,KeyError,OverflowError) as exc:result={'ok':False,'error':str(exc)}
            except ImportError:result={'ok':False,'error':'服务器视频编码组件尚未准备，请稍后重试。'}
            except (OSError,RuntimeError):result={'ok':False,'error':'声道计算失败，请重新打开工作区后重试。'}
            result['request_id']=message['request_id']
            with write_lock:protocol.write(json.dumps(result,ensure_ascii=True,allow_nan=False,separators=(',',':'))+'\n')
        except (BrokenPipeError,OSError):pass
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                raw=sys.stdin.buffer.readline(9_000_001)
                if not raw or len(raw)>9_000_000:break
                message=json.loads(raw)
                if message['op']=='video/cancel' and video is not None and message.get('body',{}).get('id')==video.id:
                    video.cancelled.set()
                if message['op'] in ('animation/stop','deactivate','heartbeat','status') or message['op']=='live' and not message.get('body',{}).get('active'):
                    dispatch(message)
                else:executor.submit(dispatch,message)
    finally:
        runtime.close()
        if video_scratch:video_scratch.cleanup()
        protocol.close()


if __name__=='__main__':main()
