"""Versioned private JSON channel. EOF and parent death end this worker."""
import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from .process_guard import watch_parent

MAX_MESSAGE=8_000_000


def main():
    config=json.loads(sys.stdin.readline(MAX_MESSAGE))
    watch_parent(config['parent_pid'])
    output_lock=threading.Lock()
    def send(data):
        with output_lock:
            sys.stdout.write(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');sys.stdout.flush()
    try:
        from .runtime import Runtime
        runtime=Runtime(config['resources'],config['profile'],playback_allowed=config.get('playback_allowed',True),legacy_profile=config.get('legacy_profile'))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({'ready':None,'error':str(exc)})
        return 1
    send({'ready':'m10/1'})
    def work(request):
        try:send({'id':request['id'],'ok':True,'value':runtime.invoke(request['op'],request.get('body'),generation=request.get('generation'))})
        except Exception as exc:send({'id':request.get('id'),'ok':False,'error':str(exc)})
    executor=ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            line=sys.stdin.readline(MAX_MESSAGE+1)
            if not line:break
            if len(line)>MAX_MESSAGE:raise ValueError('M10 message too large')
            request=json.loads(line)
            if request.get('op')=='shutdown':break
            request['generation']=runtime.cancel_generation
            if request.get('op') in ('heartbeat','status','animation/stop','deactivate') or (request.get('op')=='live' and not request.get('body',{}).get('active')):
                work(request)
            else:executor.submit(work,request)
    finally:
        runtime.cancel.set();runtime.live.playback_allowed=False;runtime.live.stop()
        executor.shutdown(wait=True,cancel_futures=True);runtime.close()


if __name__=='__main__':raise SystemExit(main())
