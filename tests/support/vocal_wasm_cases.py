"""Synthetic, shared recipes for independent native/Wasm comparisons.

Expected values are produced by the current native engine, never by Wasm.
No user speech or user profiles are read. PCM remains in scratch, not WAV files.
"""
import base64
import ctypes
import json
import time
import numpy as np
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.animation import prepare_animation


def pcm(values):
    a = np.asarray(values, dtype='<f8')
    return {'samples': len(a), 'base64': base64.b64encode(a.tobytes()).decode(),
            'rms': float(np.sqrt(np.mean(a*a))), 'peak': float(np.max(np.abs(a)))}


def run(resources):
    start = time.perf_counter()
    e = Engine(resource_dir=resources)
    e.enable_web_sections()
    output = {'snapshots': {}, 'tubes': {}, 'transfer_raw': {}, 'audio': {}, 'timings': {}, 'sample_rate': e.sr,
              'geometry_method': e.geometry_method, 'parameters': e.metadata()['parameters']}
    try:
        # Closing the analysis instance must not close synthesis or geometry.
        synth_address = ctypes.cast(e.lib.vtlInitialize, ctypes.c_void_p).value
        analysis_address = ctypes.cast(e.analysis.vtlInitialize, ctypes.c_void_p).value
        assert synth_address != analysis_address
        assert e.analysis.vtlClose() == 0
        e.reset_tube(e.prepare_tube(e.params), e.glottis(150))
        assert np.isfinite(e.block_tube(e.prepare_tube(e.params), e.glottis(150), 960)).all()
        assert e.analysis.vtlInitialize(str(e.speaker_path).encode()) == 0
        output['isolated_libraries'] = True
        for count in (33, 65):
            e.set_section_count(count)
            cases = [(name, {}) for name in ('a', 'i', 'u', 'l', 'n', 't')]
            cases += [('a', {'tongue_shape': [.15, .3, -.1, .1]}),
                      ('i', {'larynx_height': .35}), ('u', {'lip_width': .8})]
            for index, (name, options) in enumerate(cases):
                key = f'{count}/{index}/{name}'
                before = time.perf_counter()
                state = e.snapshot(e.presets[name], **options)
                output['timings'][key] = (time.perf_counter() - before) * 1000
                state.pop('compute_ms')
                output['snapshots'][key] = state
                from phonetic_core.vocal_tract.engine import dp
                mag, phase = np.empty(4096), np.empty(4096)
                assert e.geom.p2_transfer(4096, dp(mag), dp(phase)) == 0
                output['transfer_raw'][key] = {'magnitude': mag[:513].tolist(), 'phase': phase[:513].tolist()}
                tube = e.prepare_tube(e.presets[name], **options)
                output['tubes'][key] = {key: value.tolist() for key, value in tube.items()}
            for name, source in [('a', {'mode': 'voiced'}), ('s', {'mode': 'voiceless'}),
                                 ('u', {'mode': 'whisper'})]:
                frame = dict(params=e.presets[name], duration=.5, f0=150, lip_width=1., source=source)
                before = time.perf_counter()
                result = prepare_animation(e, [frame], pictures_enabled=False)
                output['timings'][f'{count}/audio/{name}'] = (time.perf_counter()-before)*1000
                output['audio'][f'{count}/{name}'] = pcm(result['audio'])
            frames = [dict(params=e.presets['a'], duration=.12, f0=150, lip_width=1.),
                      dict(aspiration=True, duration=.01, name='ʰ'),
                      dict(params=e.presets['i'], duration=.12, f0=180, lip_width=1.),
                      dict(params=e.presets['u'], duration=.05, f0=160, lip_width=1., silent=True)]
            before = time.perf_counter()
            animation = prepare_animation(e, frames, pitch_curve=[[0, 140], [1, 190]])
            output['timings'][f'{count}/sequence'] = (time.perf_counter()-before)*1000
            output['audio'][f'{count}/sequence'] = pcm(animation['audio'])
            output[f'{count}/sequence'] = {'frames': animation['frames'], 'times': animation['times'],
                                          'duration': animation['duration'], 'envelope': animation['envelope'].tolist()}
        # A diagnostic speaker copy sets its existing Jitter parameter to zero.
        # Production JD2.speaker and the production RNG/formulas are unchanged.
        import re
        import uuid
        original = e.speaker_path.read_text('utf-8')
        diagnostic, changes = re.subn(r'(<param[^>]+abbr="jitter"[^>]+value=")[^"]+', r'\g<1>0.00000', original)
        assert changes == 1
        diagnostic_path = e.speaker_path.parent / ('.ptb-m10-wasm-qa-' + uuid.uuid4().hex + '.speaker')
        with diagnostic_path.open('x', encoding='utf-8') as handle:
            handle.write(diagnostic)
        try:
            assert e.lib.vtlClose() == 0
            assert e.lib.vtlInitialize(str(diagnostic_path).encode()) == 0
            for name in ('a', 'i', 'u'):
                frame = dict(params=e.presets[name], duration=.5, f0=150, lip_width=1.)
                result = prepare_animation(e, [frame], pictures_enabled=False)
                output['audio']['diagnostic-zero-jitter/' + name] = pcm(result['audio'])
        finally:
            # Exact uniquely created file, no recursive cleanup or source edits.
            diagnostic_path.unlink()
        output['total_ms'] = (time.perf_counter()-start)*1000
        return output
    finally:
        e.close()


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(args.resources), ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
