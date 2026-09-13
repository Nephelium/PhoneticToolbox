"""M04 bounded, owned-process direct-autocorrelation timing (Windows)."""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def child(samples, order):
    import numpy as np
    from phonetic_core.lpc._legacy import compute_lpc_spectrum
    y = np.random.default_rng(404).normal(size=samples)
    start = time.perf_counter()
    f, db = compute_lpc_spectrum(y, 48000, order)
    elapsed = time.perf_counter() - start
    class Counters(c.Structure):
        _fields_ = [('cb', c.c_ulong), ('faults', c.c_ulong)] + [
            (k, c.c_size_t) for k in ('peak_ws', 'ws', 'peak_pp', 'pp',
                                    'peak_np', 'np', 'pagefile', 'peak_pagefile', 'private')]
    info = Counters()
    info.cb = c.sizeof(info)
    kernel = c.WinDLL('kernel32')
    kernel.GetCurrentProcess.restype = c.c_void_p
    get = c.WinDLL('psapi').GetProcessMemoryInfo
    get.argtypes = [c.c_void_p, c.c_void_p, c.c_ulong]
    if not get(kernel.GetCurrentProcess(), c.byref(info), info.cb):
        raise OSError('Cannot read process counters')
    print(json.dumps(dict(status='ok', elapsed=elapsed, peak_working_set=info.peak_ws,
                         peak_commit=info.peak_pagefile, finite=bool(np.isfinite(db).all()), points=len(f))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=int)
    parser.add_argument('--order', type=int, default=50)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.child is not None:
        child(args.child, args.order)
        return
    if args.output is None:
        parser.error('--output required')
    reports = []
    for order in (50, 200):
        for samples in (8000, 24000, 48000, 96000, 192000):
            command = [sys.executable, '-X', 'utf8', __file__, '--child', str(samples), '--order', str(order)]
            start = time.perf_counter()
            try:
                result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                        timeout=15, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode:
                    raise RuntimeError(result.stderr)
                report = json.loads(result.stdout)
            except subprocess.TimeoutExpired:
                # subprocess.run kills and waits only for its own child on timeout.
                report = dict(status='timeout', timeout_seconds=15)
            report.update(samples=samples, order=order, process_elapsed=time.perf_counter()-start)
            reports.append(report)
            print(json.dumps(report), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(python=sys.executable, reports=reports), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
