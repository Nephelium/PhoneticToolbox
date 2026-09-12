"""Real private worker, no device output; independent temporary profile."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import pytest
from ptb_desktop.vocal_tract.client import VocalTractClient

ROOT=Path(__file__).resolve().parents[2]

def test_worker_persistence_cancel_restart_and_owned_shutdown(tmp_path):
    c=VocalTractClient(ROOT/'resources/vocal_tract/native',tmp_path,playback_allowed=False)
    try:
        meta=c.invoke('meta');p=meta['presets']['l']
        f={'params':p,'duration':.15,'f0':125,'lip_width':1,'name':'/l/ 录制','manual_root':False}
        assert c.invoke('keyframes',{'frames':[f]*50})['saved']==50
        assert len(c.invoke('keyframes/load')['frames'])==50
        state=c.invoke('pose',{'params':p,'revision':1,'section':109})
        assert state['oral_min_area']>.1
        with pytest.raises(RuntimeError,match='过期'):c.invoke('pose',{'params':p,'revision':1})
        with ThreadPoolExecutor(2) as pool:
            job=pool.submit(c.invoke,'animation/prepare',{'frames':[dict(f,duration=3)]*4})
            time.sleep(.15);c.invoke('animation/stop')
            with pytest.raises(RuntimeError,match='取消'):job.result(timeout=10)
        process=c.process;c.close();assert process.poll() is not None
        assert c.invoke('keyframes/load')['frames'][0]['name']=='/l/ 录制'
        with pytest.raises(RuntimeError,match='禁止'):c.invoke('live',{'active':True})
    finally:c.close()

def test_queued_render_cannot_revive_after_stop(tmp_path):
    from ptb_desktop.vocal_tract.runtime import Runtime
    r=Runtime(ROOT/'resources/vocal_tract/native',tmp_path,playback_allowed=False)
    try:
        queued=r.cancel_generation;r.invoke('deactivate')
        with pytest.raises(ValueError,match='排队'):r.invoke('preview',{},generation=queued)
        assert r.live.active is False
    finally:r.close()


def test_two_windows_do_not_share_native_state_or_profile(tmp_path):
    clients=[VocalTractClient(ROOT/'resources/vocal_tract/native',tmp_path/str(i),playback_allowed=False) for i in range(2)]
    try:
        with ThreadPoolExecutor(2) as pool:
            metadata=list(pool.map(lambda c:c.invoke('meta'),clients))
        assert clients[0].process.pid!=clients[1].process.pid
        frame={'params':metadata[0]['presets']['l'],'name':'first window','duration':.2}
        clients[0].invoke('keyframes',{'frames':[frame]})
        assert clients[1].invoke('keyframes/load')['frames']==[]
        a=clients[0].invoke('pose',{'params':metadata[0]['presets']['l'],'revision':1})
        b=clients[1].invoke('pose',{'params':metadata[1]['presets']['t'],'revision':1})
        assert a['oral_min_area']>.1 and b['oral_min_area']<.001
    finally:
        for client in clients:client.close()
