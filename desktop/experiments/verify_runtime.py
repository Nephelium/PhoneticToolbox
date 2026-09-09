"""Verify owned-process lifecycle and optional WASAPI output; never saves mixed audio."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def run(command, output, count=1, loopback=False, scale=None):
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if scale:
        environment['QT_SCALE_FACTOR'] = str(scale)
    streams, chunks, recorder, device = [], [], None, None
    if loopback:
        import pyaudiowpatch as pa
        recorder = pa.PyAudio()
        device = recorder.get_default_wasapi_loopback()

        def callback(data, frames, time_info, status):
            chunks.append(data)
            return None, pa.paContinue

        stream = recorder.open(format=pa.paFloat32, channels=device['maxInputChannels'],
            rate=int(device['defaultSampleRate']), input=True, input_device_index=device['index'],
            frames_per_buffer=1024, stream_callback=callback)
        streams.append(stream)
    processes, handles, tracked = [], [], {}
    started = time.monotonic()
    failure = None
    try:
        for index in range(count):
            log = (output / f'process-{index}.log').open('wb')
            handles.append(log)
            process = subprocess.Popen([*command, '--self-test', str(output / f'instance-{index}')],
                cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            processes.append(process)
        while any(p.poll() is None for p in processes):
            for process in processes:
                try:
                    root = psutil.Process(process.pid)
                    for child in [root, *root.children(recursive=True)]:
                        tracked[(child.pid, child.create_time())] = child.name()
                except psutil.NoSuchProcess:
                    pass
            if time.monotonic() - started > 90:
                raise TimeoutError('Owned probe exceeded 90 seconds')
            time.sleep(0.1)
        time.sleep(0.6)
    except (TimeoutError, OSError) as error:
        failure = str(error)
    finally:
        for stream in streams:
            stream.stop_stream()
            stream.close()
        if recorder:
            recorder.terminate()
        # Only processes started by this exact run, never generic name/port matching.
        for process in processes:
            if process.poll() is None:
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                    for child in reversed(children):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    # Give the onefile parent a chance to collect its temporary directory.
                    process.wait(timeout=5)
                except (psutil.NoSuchProcess, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)
        for handle in handles:
            handle.close()
    alive = []
    for (pid, created), name in tracked.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created and process.is_running():
                alive.append({'pid': pid, 'name': name})
        except psutil.NoSuchProcess:
            pass
    reports = []
    for i in range(count):
        path = output / f'instance-{i}/host-report.json'
        reports.append(json.loads(path.read_text('utf-8')) if path.is_file() else
                       {'success': False, 'frozen': False, 'runtime_root': '', 'error': 'No host report'})
    result = {'instances': count, 'exit_codes': [p.returncode for p in processes],
              'tracked_process_count': len(tracked), 'owned_processes_remaining': alive,
              'wall_seconds': round(time.monotonic() - started, 3), 'scale_requested': scale,
              'host_success': [r.get('success', False) for r in reports], 'failure': failure,
              'loaded_seconds': [r.get('loaded_seconds') for r in reports],
              'onefile_temp_cleaned': [not Path(r['runtime_root']).exists() for r in reports if r['frozen']],
              'runtime_roots_distinct': len({r['runtime_root'] for r in reports}) == count if reports[0]['frozen'] else None}
    if loopback:
        import numpy as np
        data = np.frombuffer(b''.join(chunks), dtype=np.float32).reshape(-1, device['maxInputChannels'])
        rate = int(device['defaultSampleRate'])
        window = 8192
        frequencies = np.fft.rfftfreq(window, 1 / rate)
        metrics = []
        for channel, expected in enumerate([440, 660]):
            peaks, contrasts = [], []
            for start in range(0, len(data) - window, window // 4):
                spectrum = abs(np.fft.rfft(data[start:start + window, channel] * np.hanning(window)))
                signal = float(max(spectrum[abs(frequencies - expected) < 12]))
                floor = float(np.median(spectrum[(frequencies > 300) & (frequencies < 1500)])) + 1e-12
                peaks.append(signal)
                contrasts.append(signal / floor)
            # Search for a stable tone window, not the loudest transient in the mix.
            candidates = [i for i, peak in enumerate(peaks) if peak > 0.05]
            best = max(candidates, key=lambda i: contrasts[i]) if candidates else 0
            metrics.append({'channel': channel + 1, 'expected_hz': expected,
                            'peak_spectrum': peaks[best] if peaks else 0,
                            'spectral_contrast': contrasts[best] if contrasts else 0,
                            'loudest_window_contrast': contrasts[int(np.argmax(peaks))] if peaks else 0,
                            'stable_window_start_seconds': best * (window // 4) / rate})
        result['loopback'] = {'device': device['name'], 'sample_rate_hz': rate,
                              'captured_frames': len(data), 'raw_audio_saved': False, 'metrics': metrics,
                              'test_tones_detected': all(m['peak_spectrum'] > 0.05 and m['spectral_contrast'] > 30 for m in metrics),
                              'scope': 'WASAPI digital render endpoint; not a microphone recording or acoustic latency measurement'}
        chunks.clear()
    result['success'] = (all(code == 0 for code in result['exit_codes']) and all(result['host_success'])
        and not alive and all(result['onefile_temp_cleaned'])
        and (not loopback or result['loopback']['test_tones_detected']))
    (output / 'runtime-report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result['success']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instances', type=int, choices=[1, 2], default=1)
    parser.add_argument('--loopback', action='store_true')
    parser.add_argument('--scale', type=float)
    args = parser.parse_args()
    command = [str(args.exe.resolve())] if args.exe else [sys.executable, '-X', 'utf8', str(Path(__file__).with_name('host_probe.py'))]
    sys.exit(0 if run(command, args.output.resolve(), args.instances, args.loopback, args.scale) else 1)
