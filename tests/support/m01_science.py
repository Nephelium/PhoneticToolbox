"""Synthetic Windows numerical probe ONLY, not a production native adapter.

The exact inherited executable is used with tiny generated inputs in pytest's
owned directory. M01-C must implement/enforce resource accounting before any
server/user task can execute REAPER. No imports from the legacy Python package.
"""
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from phonetic_core.acoustic.reaper_codec import reaper_pcm16, parse_est_f0
from phonetic_core.models.associations import AcousticAssociations, Interval, Tier
from phonetic_core.ports.acoustic import ReaperTrack, python_reaper

ROOT = Path(__file__).resolve().parents[2]


def native_probe(folder, fault=False):
    calls = []
    def run(audio, frame_interval_sec, min_f0, max_f0, *, hilbert, no_highpass):
        assert len(audio.samples) <= 64000, 'test probe accepts small synthetic signals only'
        binary = ROOT/'phonetic_toolbox/core/acoustic/reaper.exe'
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == '279fecc82ed0a49b0277b114270771d7670299068e849058b392672825981824'
        source, destination = folder/'input_16k.wav', folder/'out.f0'
        wavfile.write(source, 16000, reaper_pcm16(audio))
        argv = [str(binary), '-i', str(source), '-f', str(destination), '-a',
                '-e', str(float(frame_interval_sec)), '-m', str(float(min_f0)), '-x', str(float(max_f0))]
        if hilbert: argv.append('-t')
        if no_highpass: argv.append('-s')
        if fault: argv = [str(binary), '--m01-invalid-option']
        completed = subprocess.run(argv, cwd=folder, capture_output=True, timeout=45,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        calls.append(completed.returncode)
        if completed.returncode:
            if not fault: raise AssertionError('unexpected native probe failure')
            track = python_reaper(audio, frame_interval_sec, min_f0, max_f0,
                                   hilbert=hilbert, no_highpass=no_highpass)
            return ReaperTrack(track.times, track.values, track.actual_backend, 'native_exit_1')
        times, _, values = parse_est_f0(destination.read_text('utf-8'))
        return ReaperTrack(times, np.where(values > 0, values, np.nan), 'native_reaper_test_probe')
    run.calls = calls
    return run


def associations(formula=False):
    times = [.5, 0., .25, .25, .75, float('nan')]
    lip = {'metadata': {'lip_manual_offset': .125, 'audio_first_frame_time': 100.},
           'relative_times': [x+10 for x in times], 'absolute_timestamps': [x+100 for x in times]}
    for i, key in enumerate(['area', 'outer_width', 'open', 'circularity'], 1):
        lip[key] = [3*i, 1*i, 2*i, 99*i, 4*i, 5*i]
    lip['area'][2] = float('nan')
    tiers = (Tier('音节', (Interval(0., .8, '测试'),)), Tier('IPA',
        (Interval(0., .2, 'aː'), Interval(.2, .4, 'iː'), Interval(.4, .8, '=literal' if formula else '标签'))))
    return AcousticAssociations(lip=lip, tiers=tiers)


def fail_irapt(*args, **kwargs):
    raise RuntimeError('Controlled IRAPT unavailability')


def pack(value):
    a = np.asarray(value)
    result = {'values': a.tolist(), 'shape': list(a.shape), 'dtype': str(a.dtype)}
    if a.dtype.kind in 'fiu':
        result['nonfinite'] = np.where(np.isnan(a), 1, np.where(np.isposinf(a), 2, np.where(np.isneginf(a), 3, 0))).tolist()
        result['values'] = np.where(np.isfinite(a), a, None).tolist()
    return result
