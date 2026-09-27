"""P11 controlled numeric diagnostics; never an alternative acceptance test.

Captures installed-core outputs and explicitly labelled intermediate experiments
from public fixtures. It does not modify the core, fixtures or pytest assertions.
Linux must already be inside a bounded P11 cgroup before heavy imports.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
from contextlib import redirect_stdout


def compare_arrays(actual, expected):
    """Exact comparison plus descriptive errors, with no acceptance tolerance."""
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    if a.dtype.hasobject or b.dtype.hasobject:
        raise ValueError('Object arrays are not numeric evidence')
    result = dict(shape=list(a.shape), reference_shape=list(b.shape), dtype=str(a.dtype),
                  reference_dtype=str(b.dtype), exact=a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes())
    if a.shape != b.shape:
        return result
    result['nan_mask_equal'] = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    result['positive_inf_mask_equal'] = bool(np.array_equal(np.isposinf(a), np.isposinf(b)))
    result['negative_inf_mask_equal'] = bool(np.array_equal(np.isneginf(a), np.isneginf(b)))
    valid = np.isfinite(a) & np.isfinite(b)
    av, bv = a[valid].astype('float64'), b[valid].astype('float64')
    delta = np.abs(av-bv)
    result.update(finite_values=int(valid.sum()), changed_finite=int(np.count_nonzero(av != bv)),
                  max_abs=float(delta.max(initial=0)),
                  max_rel_nonzero=float((delta[bv != 0]/np.abs(bv[bv != 0])).max(initial=0)))
    return result


def boundary():
    if sys.platform == 'linux':
        group = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        if '..' in Path(group).parts or not any(x.startswith('ptb-p11-') for x in Path(group).parts):
            raise RuntimeError('P11 cgroup required before imports')
        folder = Path('/sys/fs/cgroup')/group.lstrip('/')
        maximum = (folder/'memory.max').read_text().strip()
        if not maximum.isdigit() or int(maximum) > 1073741824:
            raise RuntimeError('MemoryMax must be at most 1 GiB')
        return folder
    if sys.platform != 'win32':
        raise RuntimeError('Platform not in this diagnostic matrix')
    return None


def capture(root, output, label):
    cgroup = boundary()
    # Keep DLL directories local to this diagnostic child, never the host process.
    dll = Path(sys.prefix)/'Library/bin'
    dll_handle = os.add_dll_directory(str(dll)) if sys.platform == 'win32' and dll.is_dir() else None
    import numpy as np
    import scipy
    from scipy import signal, linalg
    import phonetic_core
    from phonetic_core.egg import EGGConfig, prepare, analyze_events, cq_segment, events_segment
    from phonetic_core.egg.metrics import calculate_cq_sq
    from phonetic_core.egg.f0 import praat_pitch
    from phonetic_core.egg.inverse import inverse_filter
    from phonetic_core.lpc import LPCConfig, compute_spectrum
    if not Path(phonetic_core.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Installed core wheel required')
    output.mkdir(exist_ok=False, parents=True)
    values, none_fields, errors, input_hashes = {}, [], [], {}

    def record(name, value):
        if value is None:
            none_fields.append(name)
        else:
            a = np.asarray(value)
            if a.dtype.hasobject:
                raise ValueError('Non-numeric capture '+name)
            values[name] = a.copy()

    def load(folder, filename):
        path = root/'tests/fixtures'/folder/filename
        input_hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    meta = json.loads(load('m03', 'EGG-SYN-PCM16.json').read_text('utf-8'))
    with np.load(load('m03', 'EGG-SYN-PCM16.npz'), allow_pickle=False) as source:
        old = dict(source)
    cfg = EGGConfig(**meta['variants'][0]['config'])
    samples = np.column_stack((old['load.egg_signal_raw'], old['load.audio_signal']))
    state = prepare(samples, 44100, cfg)
    for key in ('time_vector', 'egg_signal_raw', 'audio_signal', 'egg_signal_processed'):
        record('egg.prepare.'+key, getattr(state, key))
        record('reference.egg.prepare.'+key, old['load.'+key])
    # Decompose exactly the existing preparation calls; these are diagnostics only.
    detrended = signal.detrend(state.egg_signal_raw)
    record('egg.detrended', detrended)
    data = detrended
    for kind, cutoff in (('high', cfg.highpass_cutoff), ('low', cfg.lowpass_cutoff)):
        b, a = signal.butter(4, min(max(cutoff/(44100*.5), 1e-6), 1-1e-6), btype=kind)
        record('egg.filter.'+kind+'.b', b)
        record('egg.filter.'+kind+'.a', a)
        record('egg.filter.'+kind+'.zi', signal.lfilter_zi(b, a))
        data = signal.filtfilt(b, a, data)
        record('egg.filter.'+kind+'.output', data)
    for index, variant in enumerate(meta['variants']):
        config = EGGConfig(**variant['config'])
        for mode in ('actual', 'frozen_preprocessed'):
            current = dataclasses.replace(state, egg_signal_processed=old['load.egg_signal_processed'].copy()) if mode == 'frozen_preprocessed' else dataclasses.replace(state)
            current = analyze_events(current, config)
            prefix = f'egg.variant.{index}.{mode}'
            for key, data in zip(('gci', 'goi', 'peak'), (current.gci_times, current.goi_times, current.peak_times)):
                record(prefix+'.'+key, data)
            for key, data in zip(('time', 'cq', 'sq'), calculate_cq_sq(current.gci_times, current.goi_times, current.peak_times)):
                record(prefix+'.global.'+key, data)
            if mode == 'actual':
                # Continue every ROI even when a preceding field differs.
                for roi, modes in variant['roi'].items():
                    for source_mode, item in modes.items():
                        for name, fn in (('cq', cq_segment), ('events', events_segment)):
                            result = fn(current, *item['bounds_s'], config, use_raw_signal=source_mode == 'raw')
                            for n, value in enumerate(result):
                                record(prefix+f'.roi.{roi}.{source_mode}.{name}.{n}', value)
    track = praat_pitch(old['load.audio_signal'], 44100)
    for key in ('values', 'times', 'legacy_times'):
        record('praat.'+key, getattr(track, key))
    variant = next(v for v in meta['variants'] if v['config']['gci_method'] == 'slope' and v['config']['goi_method'] == 'scale' and v['config']['auto_prominence'])
    audio = old['load.audio_signal'][:int(.12*44100)]
    events = np.array([t for t in variant['events'][0] if t < len(audio)/44100])
    for order, key in ((None, 'auto'), (12, 'explicit')):
        record('inverse.'+key, inverse_filter(audio, 44100, events, lp_order=order))
    lpc_meta = json.loads(load('m04', 'result.json').read_text('utf-8'))
    with np.load(load('m04', 'arrays.npz'), allow_pickle=False) as source:
        lpc_values = dict(source)
    for name, case in lpc_meta['cases'].items():
        audio = lpc_values[name+'.input'].copy()
        try:
            result = compute_spectrum(audio, case['rate'], LPCConfig(**case['config']))
        except Exception as exc:
            errors.append(dict(case=name, type=type(exc).__name__, expected=case['status'] == 'error'))
            continue
        record('lpc.'+name+'.frequency', result.frequencies_hz)
        record('lpc.'+name+'.db', result.magnitude_db)
        record('lpc.'+name+'.yrange', [result.amp_min_db, result.amp_max_db])
        y = np.asarray(audio, dtype=np.float64).reshape(-1)
        pre = np.append(y[0], y[1:]-.97*y[:-1])
        window = np.hamming(pre.size)
        y_win = pre*window
        corr = np.correlate(y_win, y_win, mode='full');corr=corr[corr.size//2:]
        order=case['config']['order'];a=np.concatenate(([1.], -linalg.solve_toeplitz((corr[:order],corr[:order]),corr[1:order+1])))
        f,h=signal.freqz(1.,a,worN=1024,fs=case['rate'])
        for key,value in (('pre',pre),('window',window),('windowed',y_win),('correlation',corr),('coefficients',a),('response_real',h.real),('response_imag',h.imag)):
            record('lpc.stage.'+name+'.'+key,value)
    capture_file=output/'arrays.npz'
    np.savez_compressed(capture_file, **values)
    config=io.StringIO()
    with redirect_stdout(config):np.show_config();scipy.show_config()
    packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','praat-parselmouth','phonetic-core')}
    library_hashes={}
    if sys.platform == 'linux':
        for line in Path('/proc/self/maps').read_text().splitlines():
            path=line.split()[-1]
            if path.startswith('/') and any(s in Path(path).name.lower() for s in ('blas','lapack','mkl')):
                library_hashes[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    info=dict(label=label,platform=platform.platform(),python=sys.version,executable=sys.executable,
              packages=packages,imports={m.__name__:m.__file__ for m in (np,scipy,phonetic_core)},
              thread_settings={key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','MKL_CBWR','OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES')},
              core_sources={f.relative_to(Path(phonetic_core.__file__).parent).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
                            for folder in ('egg','lpc') for f in (Path(phonetic_core.__file__).parent/folder).glob('*.py')},
              input_hashes=input_hashes,array_file_sha256=hashlib.sha256(capture_file.read_bytes()).hexdigest(),
              array_hashes={key:hashlib.sha256(a.tobytes()).hexdigest() for key,a in values.items()},
              none_fields=none_fields,errors=errors,build_config=config.getvalue(),library_hashes=library_hashes,
              cgroup={name:(cgroup/name).read_text().strip() for name in ('memory.max','cpu.max','pids.max','memory.peak')} if cgroup else None)
    (output/'capture.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(label=label,arrays=len(values),none_fields=len(none_fields),errors=errors)))
    if dll_handle:dll_handle.close()


def compare(actual, reference, output):
    import numpy as np
    am=json.loads((actual/'capture.json').read_text('utf-8'));bm=json.loads((reference/'capture.json').read_text('utf-8'))
    # Old diagnostic captures used host-native separators. Normalize identifiers,
    # never hashes or file contents, when comparing Windows and Linux evidence.
    if normalized_hashes(am['input_hashes']) != normalized_hashes(bm['input_hashes']):raise ValueError('Input hashes differ')
    with np.load(actual/'arrays.npz',allow_pickle=False) as x,np.load(reference/'arrays.npz',allow_pickle=False) as y:
        if set(x.files) != set(y.files):raise ValueError('Captured field sets differ')
        records={key:compare_arrays(x[key],y[key]) for key in x.files}
    result=dict(actual=am['label'],reference=bm['label'],none_fields_equal=am['none_fields']==bm['none_fields'],
                errors_equal=am['errors']==bm['errors'],exact=sum(v['exact'] for v in records.values()),
                different=sum(not v['exact'] for v in records.values()),fields=records)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('actual','reference','exact','different')}))


def normalized_hashes(values):
    normalized={key.replace('\\', '/'): value for key,value in values.items()}
    if len(normalized) != len(values):raise ValueError('Ambiguous input paths')
    return normalized


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    c=sub.add_parser('capture');c.add_argument('--root',type=Path,required=True);c.add_argument('--output',type=Path,required=True);c.add_argument('--label',required=True)
    c=sub.add_parser('compare');c.add_argument('--actual',type=Path,required=True);c.add_argument('--reference',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.action=='capture':capture(a.root,a.output,a.label)
    else:compare(a.actual,a.reference,a.output)


if __name__=='__main__':main()
