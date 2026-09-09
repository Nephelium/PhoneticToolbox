"""Run only in the original v2 reference interpreter, never in the v3 application."""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import sys
import warnings
from pathlib import Path

from baseline_support import sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text('utf-8'))
    source = Path(request['source_root']).resolve()
    wav = Path(request['input']).resolve()
    destination = Path(request['output']).resolve()
    assert destination.is_relative_to(Path(request['output_root']).resolve())
    assert not destination.is_relative_to(source)
    assert sys.dont_write_bytecode, 'Reference source must remain read-only'
    sys.path.insert(0, str(source))
    import numpy as np
    import scipy.io.wavfile as wavfile
    import phonetic_toolbox
    assert Path(phonetic_toolbox.__file__).resolve().is_relative_to(source)
    from phonetic_toolbox.models.config import AcousticConfig, EGGConfig
    from phonetic_toolbox.services.acoustic_service import AcousticAnalysisService, PARAMETER_MAPPING
    from phonetic_toolbox.services.egg_service import EGGAnalysisService

    def pack(value):
        if dataclasses.is_dataclass(value):
            return {f.name: pack(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, np.ndarray):
            value = np.asarray(value)
            if value.dtype.kind in 'fiu':
                finite = np.isfinite(value)
                mask = np.where(np.isnan(value), 1, np.where(np.isposinf(value), 2, np.where(np.isneginf(value), 3, 0)))
                result = {'shape': list(value.shape), 'dtype': str(value.dtype)}
                if value.size > 10000:
                    result.update(storage='digest', sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest(),
                        finite_count=int(finite.sum()), nonfinite_counts=[int((mask == n).sum()) for n in range(4)],
                        first=pack(value.flat[0]), last=pack(value.flat[-1]))
                else:
                    result.update(storage='values', values=np.where(finite, value, None).tolist(), nonfinite=mask.tolist())
                return result
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'storage': 'values', 'values': value.tolist()}
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, (tuple, list)):
            return [pack(v) for v in value]
        if isinstance(value, dict):
            return {str(k): pack(v) for k, v in value.items()}
        return value

    calls, returns, processes = set(), {}, []
    observed = {'compute_praat_f0_track', 'compute_reaper_f0', 'irapt', '_extract_f0_for_wm',
                'compute_silence_mask', 'compute_voiced_mask'}

    def profile(frame, event, value):
        module = frame.f_globals.get('__name__', '')
        name = frame.f_code.co_name
        if event == 'call' and module.startswith('phonetic_toolbox'):
            calls.add(module + ':' + name)
        if event == 'return' and module.startswith('phonetic_toolbox') and name in observed:
            returns[name] = pack(value)
        if event == 'return' and module == 'subprocess' and name == 'run' and value is not None:
            argv = value.args
            executable = Path(argv[0])
            processes.append({'executable_name': executable.name, 'executable_sha256': sha(executable),
                'returncode': value.returncode,
                'arguments': [str(a) if not (':' in str(a) or '\\' in str(a) or '/' in str(a)) else '<path>' for a in argv[1:]]})

    messages = []
    class CaptureLog(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage().replace(str(wav), '<input>').replace(str(source), '<v2>'))
    handler = CaptureLog()
    logging.getLogger('phonetic_toolbox').addHandler(handler)
    config = AcousticConfig() if request['mode'] == 'acoustic' else EGGConfig()
    if request['mode'] == 'acoustic':
        config.reaper_bin_path = str(source / 'phonetic_toolbox/core/acoustic/reaper.exe')
    for key, value in request.get('config_overrides', {}).items():
        setattr(config, key, value)
    scientific = {'mode': request['mode'], 'config': dataclasses.asdict(config)}
    if 'reaper_bin_path' in scientific['config']:
        scientific['config']['reaper_bin_path'] = '<v2>/phonetic_toolbox/core/acoustic/reaper.exe'
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            fs, samples = wavfile.read(wav)
            scientific['audio'] = {'sample_rate_hz': int(fs), 'sample_count': len(samples),
                'channels': 1 if samples.ndim == 1 else samples.shape[1], 'dtype': str(samples.dtype)}
            sys.setprofile(profile)
            if request['mode'] == 'acoustic':
                service = AcousticAnalysisService()
                result = service.analyze_file(str(wav), config, textgrid_path=request.get('textgrid'))
                scientific['result'] = pack(result)
                frame = result.to_dataframe()
                scientific['dataframe_columns'] = list(frame.columns)
                scientific['parameter_mapping'] = PARAMETER_MAPPING
            else:
                service = EGGAnalysisService()
                flip = request['flip_channels']
                result = service.load_file(str(wav), config, flip_channels=flip)
                result = service.analyze_events(result, config)
                scientific['channel_selection'] = {'flip_channels': flip, 'evidence': request['channel_evidence']}
                scientific['result'] = pack(result)
                scientific['cq_sq_roi'] = pack(service.calculate_cq_sq_segment(result, 0, min(0.5, result.file_duration), config))
            scientific['status'] = 'returned'
        except Exception as exc:
            scientific['status'] = 'error'
            scientific['error_type'] = type(exc).__name__
            scientific['error_message'] = str(exc).replace(str(wav), '<input>').replace(str(source), '<v2>')
        finally:
            sys.setprofile(None)
    scientific['lowlevel_returns'] = returns
    scientific['observed_calls'] = sorted(calls)
    scientific['native_processes'] = processes
    scientific['warnings'] = sorted(set(str(w.message).replace(str(wav), '<input>').replace(str(source), '<v2>') for w in captured))
    scientific['log_messages'] = messages
    modules = {}
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if name.startswith('phonetic_toolbox') and file:
            path = Path(file).resolve()
            assert path.is_relative_to(source), 'Unexpected reference module: ' + name
            modules[name] = sha(path)
    dependencies = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    output = {'schema_version': 1, 'case_id': request['case_id'], 'input_sha256': sha(wav),
              'producer': {'root_kind': 'original_v2', 'python': sys.version,
                           'modules_sha256': modules, 'installed_dependencies': dependencies},
              'scientific': scientific}
    write_json(destination, output)
    print(json.dumps({'case_id': request['case_id'], 'status': scientific['status'],
                      'actual_native_runs': len(processes)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
