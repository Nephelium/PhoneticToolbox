"""P11 diagnostic comparison against unchanged public Windows/V2 fixtures.

Run inside an explicitly bounded Linux systemd unit. Differences are evidence,
never a relaxed acceptance criterion. No private audio or service database.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Native Linux required')
    group = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    if '..' in Path(group).parts or 'ptb-p11-' not in group:
        parser.error('Explicit P11 cgroup required')
    cgroup = Path('/sys/fs/cgroup') / group.lstrip('/')
    maximum = (cgroup/'memory.max').read_text().strip()
    if not maximum.isdigit() or int(maximum) > 1073741824:
        parser.error('Group memory must be bounded to at most 1 GiB')
    if args.output.exists():
        parser.error('Evidence already exists')
    import numpy as np
    import scipy
    import phonetic_core
    if not Path(phonetic_core.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Installed wheel required')
    records, failures = [], []
    label = ''

    def compare(actual, expected):
        a, b = np.asarray(actual), np.asarray(expected)
        item = dict(case=label, index=len(records), shape=list(a.shape), expected_shape=list(b.shape),
                    dtype=str(a.dtype), expected_dtype=str(b.dtype))
        # Object arrays (notably scalar None) contain process-local pointers.
        # Serialize their values; their raw memory cannot be a repeatability hash.
        actual_bytes = json.dumps(a.tolist(), ensure_ascii=False, sort_keys=True).encode() if a.dtype.hasobject else a.tobytes()
        expected_bytes = json.dumps(b.tolist(), ensure_ascii=False, sort_keys=True).encode() if b.dtype.hasobject else b.tobytes()
        item['exact'] = a.shape == b.shape and a.dtype == b.dtype and actual_bytes == expected_bytes
        item['actual_sha256'] = hashlib.sha256(actual_bytes).hexdigest()
        if a.shape == b.shape and a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu':
            item['nan_mask_equal'] = bool(np.array_equal(np.isnan(a), np.isnan(b)))
            item['inf_mask_equal'] = bool(np.array_equal(np.isinf(a), np.isinf(b)))
            valid = np.isfinite(a) & np.isfinite(b)
            x, y = a[valid].astype('float64'), b[valid].astype('float64')
            diff = np.abs(x-y)
            item['finite_values'] = int(valid.sum())
            item['max_abs'] = float(diff.max(initial=0))
            nonzero = y != 0
            item['max_rel_nonzero'] = float((diff[nonzero]/np.abs(y[nonzero])).max(initial=0))
        records.append(item)

    def load(name):
        path = args.root/'tests/parity'/('test_'+name+'.py')
        spec = importlib.util.spec_from_file_location('p11_'+name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def invoke(name, fn, *values):
        nonlocal label
        label = name
        try:
            fn(*values)
        except Exception as exc:
            failures.append(dict(case=name, type=type(exc).__name__, detail=str(exc)[:600]))

    egg = load('egg_analysis')
    reference = egg.reference.__wrapped__()
    egg.equal = compare
    for index in range(8):
        invoke('M03.variant.'+str(index), egg.test_methods_thresholds_global_and_both_roi_policies, reference, index)
    invoke('M03.praat', egg.test_praat_array_input_matches_actual_values_and_legacy_time, reference)
    for order, key in ((None, 'auto'), (12, 'explicit')):
        invoke('M03.inverse.'+key, egg.test_inverse_array_output, reference, order, key)
    lpc = load('lpc_spectrum')
    lpc.identical = compare
    arrays = lpc.arrays.__wrapped__()
    for name in lpc.META['cases']:
        invoke('M04.'+name, lpc.test_original_spectrum, name, arrays)
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
        scipy.show_config()
    libraries = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        value = line.split()[-1]
        if value.startswith('/') and any(key in value.lower() for key in ('blas', 'lapack', 'mkl')):
            path = Path(value)
            libraries[value] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {}
    for name in ('numpy', 'scipy', 'praat-parselmouth', 'pandas', 'matplotlib', 'phonetic-core'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(platform=platform.platform(), python=sys.version, executable=sys.executable,
                  core_import=phonetic_core.__file__, packages=versions,
                  build_config=config.getvalue(), libraries=libraries,
                  resource_limits={name:(cgroup/name).read_text().strip() for name in
                  ('memory.max', 'memory.swap.max', 'cpu.max', 'pids.max', 'memory.peak')},
                  comparisons=records, assertion_failures=failures,
                  exact_passed=sum(r['exact'] for r in records), exact_failed=sum(not r['exact'] for r in records))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key:result[key] for key in ('exact_passed','exact_failed')}))
    return 2 if result['exact_failed'] or failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
