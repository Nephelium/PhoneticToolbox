"""The same native-state checks run on Windows/Linux and the real Wasm ABI."""
import threading
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.session import Runtime
from phonetic_core.vocal_tract.animation import prepare_animation
from ptb_worker.vocal_web import BrowserAudio, MemoryProfile


def run(resources):
    output=[]
    for speaker in ('JD2','W02'):
        engine=Engine(speaker,resource_dir=resources)
        try:
            runtime=Runtime(engine,MemoryProfile(),BrowserAudio(engine))
            runtime.invoke('pose',dict(params=engine.presets['a'],revision=1,f0=240.,manual_root=False))
            frame=dict(params=engine.presets['i'],f0=240.,lip_width=1.,duration=.1,manual_root=True)
            prepare_animation(engine,[frame])
            assert engine.manual_root is False
            engine.set_manual_root(True)
            runtime.invoke('live',{'active':True})
            assert engine.manual_root is False
            runtime.invoke('pose',dict(params=engine.presets['i'],revision=2,f0=240.,manual_root=False))
            expected=engine.pitch_level_gain(runtime.live.tube,240.,runtime.live.source)
            assert runtime.live.pitch_gain==expected
            runtime.live.stop()
            original=engine.block_tube
            for outcome in ('cancel','exception'):
                cancel=threading.Event()
                def interrupt(tube,glottis,n=960):
                    if n and outcome=='exception':raise RuntimeError('controlled native failure')
                    result=original(tube,glottis,n)
                    if n:cancel.set()
                    return result
                engine.block_tube=interrupt
                try:
                    prepare_animation(engine,[frame],cancel=cancel)
                except (ValueError,RuntimeError) as error:
                    assert ('取消' in str(error)) if outcome=='cancel' else str(error)=='controlled native failure'
                else:raise AssertionError('render did not stop')
                assert engine.manual_root is False
            engine.block_tube=original
            for manual in (False,True):
                engine.set_manual_root(manual);runtime.manual_root=manual
                for outcome in ('success','cache','preprocess_error','render_error','cancel'):
                    runtime.animation=None
                    preprocess=engine.constrain_nasal_opening
                    def fail_preprocess(*args):raise RuntimeError('controlled preprocessing failure')
                    def fail_block(tube,glottis,n=960):
                        if n and outcome=='render_error':raise RuntimeError('controlled rendering failure')
                        result=original(tube,glottis,n)
                        if n and outcome=='cancel':runtime.cancel.set()
                        return result
                    if outcome=='preprocess_error':engine.constrain_nasal_opening=fail_preprocess
                    engine.block_tube=fail_block
                    try:
                        body=dict(frames=[{**frame,'manual_root':not manual}],keep_vowel=True)
                        try:
                            runtime.invoke('animation/prepare',body)
                            assert outcome in ('success','cache')
                            if outcome=='cache':assert runtime.invoke('animation/prepare',body)['cached']
                        except (ValueError,RuntimeError):
                            assert outcome not in ('success','cache')
                        assert engine.manual_root is manual
                    finally:
                        engine.block_tube=original;engine.constrain_nasal_opening=preprocess
            runtime.invoke('pose',dict(params=engine.presets['a'],revision=3,f0=120.,manual_root=False))
            clip=runtime.invoke('animation/prepare',dict(frames=[frame]))
            runtime.invoke('animation/play',{'id':clip['id']});runtime.live.pull({})
            runtime.live.acknowledge({'position':.05})
            shown=runtime.invoke('animation/frame',{'id':clip['id']})['state']
            stopped=runtime.invoke('animation/stop')
            assert stopped['pose']==shown and runtime.live.f0==240.
            assert runtime.live.params.tolist()==shown['params']
            assert engine.manual_root is runtime.manual_root is True
            output.append(dict(speaker=speaker,root_restored=True,compensation_matches=True,
                               pitch_gain=expected,cancel_restored=True,exception_restored=True,
                               keep_vowel_all_exits_restored=True,stop_pose_adopted=True))
        finally:engine.close()
    return output
