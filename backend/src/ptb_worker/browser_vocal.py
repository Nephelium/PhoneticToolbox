"""Device-free browser host for the unchanged core VTL state machine."""
import json
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.session import Runtime
from .vocal_web import BrowserAudio, MemoryProfile


class BrowserVocalRuntime:
    def __init__(self, resources, checkpoint=None):
        engine = Engine(resource_dir=resources)
        options=list(engine.section_options)
        engine.enable_web_sections()
        engine.section_options=[33,*options]
        engine.set_section_count(65)
        self.runtime = Runtime(engine, MemoryProfile(), BrowserAudio(engine))
        if checkpoint:
            self.restore(checkpoint)

    def restore(self, state):
        runtime = self.runtime
        if state.get('speaker','JD2')!='JD2':runtime.invoke('speaker/select',{'speaker':state['speaker']})
        runtime.profile.models=state.get('models',{})
        if state.get('library'):
            runtime.invoke('library/save',state['library'])
        else:
            runtime.invoke('presets/save',{'presets':state['presets']})
            runtime.invoke('favorites/save',{'ids':state['favorites']})
        runtime.invoke('keyframes',state['sequence'])
        runtime.invoke('audio/settings',state['audio'])
        runtime.invoke('source/perturbation',{'perturbation':state.get('perturbation')})
        if state.get('pose'):
            runtime.invoke('pose', state['pose'])

    def checkpoint(self):
        r = self.runtime
        a = r.live
        pose = None if not r.revision else dict(params=a.params.tolist(), revision=r.revision,
                    section=r.section, section_count=r.engine.section_count, f0=a.f0,
                    lip_width=a.lip_width, larynx_height=a.larynx_height,
                    tongue_shape=a.tongue_shape, lip_shape=a.lip_shape, source=a.source, manual_root=r.manual_root)
        return dict(library=r.profile.load_library(r.engine),presets=r.profile.presets, favorites=r.profile.favorites,
                    sequence=r.profile.sequence, audio=dict(device='browser', volume=a.volume, gain_db=a.gain_db), pose=pose,
                    perturbation=r.perturbation,speaker=r.engine.speaker,models=r.profile.models)

    def invoke(self, op, body):
        if not isinstance(body, dict):
            raise ValueError('无效的声道请求')
        r = self.runtime
        if op == 'audio/pull':
            value = r.live.pull(body)
        elif op == 'shutdown':
            r.close()
            return {'closed': True}
        else:
            if op in ('animation/frame', 'audio/monitor', 'animation/stop', 'deactivate', 'live') and 'position' in body:
                r.live.acknowledge(body)
            if op in ('heartbeat', 'status') and body.get('completed') is True:
                r.live.acknowledge(body)
            value = r.invoke(op, body)
        if op == 'meta':
            value.update(continuous_audio=True, preview_duration=.5,
                         execution='browser-wasm')
        if op in ('status', 'heartbeat', 'preview', 'live', 'audio/test', 'animation/play'):
            value['playback_generation'] = r.live.output_history.generation
        return value

    def dispatch(self, op, body):
        try:
            result = {'value': self.invoke(op, json.loads(body))}
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return json.dumps({'error': str(error)}, ensure_ascii=False)
        fields = {'pose': 'pose', 'source/update':'pose','keyframes': 'sequence', 'presets/save': 'presets',
                  'favorites/save': 'favorites', 'audio/settings': 'audio','source/perturbation':'perturbation'}
        if op in ('meta','speaker/select','library/save','library/edit','library/import','presets/save','favorites/save'):
            result['checkpoint'] = self.checkpoint()
        elif op in fields:
            key = fields[op]
            result['checkpoint'] = {key: self.checkpoint()[key]}
        elif ('pose' in result['value'] or op=='animation/frame' and not result['value'].get('active',True)):
            result['checkpoint'] = {'pose': self.checkpoint()['pose']}
        return json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
