"""Synthetic Windows source-path benchmark; never opens recording hardware."""
import argparse
import hashlib
import io
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ('desktop/src', 'backend/src', 'packages/phonetic_core/src')]
from verification_artifacts import verification_run
from workbench_source import bind_sources


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def measure(call):
    values, elapsed = [], []
    for _ in range(4):
        start = time.perf_counter()
        values.append(call())
        elapsed.append((time.perf_counter() - start) * 1000)
    assert len({digest(v) for v in values}) == 1
    return dict(milliseconds=elapsed, sha256=digest(values[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--expect-optimized', action='store_true')
    args = parser.parse_args()
    bind_sources()
    import numpy as np
    import ptb_desktop.recording.service as recording
    from ptb_desktop.recording.storage import Project, save_pcm
    from ptb_worker.spectrogram_session import SpectrogramSession
    rate = 16000
    t = np.arange(rate * 10) / rate
    x = np.column_stack((.2 * np.sin(2 * np.pi * 137 * t), .3 * np.sin(2 * np.pi * 211 * t))).astype('<f4')
    wav = io.BytesIO()
    with wave.open(wav, 'wb') as output:
        output.setnchannels(2); output.setsampwidth(2); output.setframerate(rate)
        output.writeframes((x * 32767).astype('<i2').tobytes())
    with verification_run('desktop-preview-performance', args.label,
                          '10 s, 16 kHz stereo sinusoids 137/211 Hz, 4 identical previews; no device or user files') as (out, scratch):
        service, session = recording.RecordingService(), SpectrogramSession()
        counts = dict(recording_scans=0, recording_spectra=0, spectrogram_exchanges=0)
        original_iter, original_display = recording.iter_audio, recording.DisplaySpectrum
        def scanned(*a, **kw):
            counts['recording_scans'] += 1
            return original_iter(*a, **kw)
        def display(*a, **kw):
            counts['recording_spectra'] += 1
            return original_display(*a, **kw)
        try:
            service.project = Project(scratch / 'recording', True)
            span = save_pcm(service.project.root, 'audio/synthetic.pcm', x)
            data = service.project.data.copy()
            data['takes'] = [dict(id='synthetic', config=dict(sample_rate=rate, channels=2), head=0,
                                 versions=[dict(id='raw', spans=[span])])]
            service.project.commit(data)
            recording.iter_audio, recording.DisplaySpectrum = scanned, display
            view = dict(channel=0, start=0, end=10, width=900)
            # Warm library startup separately, then count actual exchanges.
            initial = session.render(wav.getvalue(), **view)
            original_write = session.process.stdin
            class Input:
                def write(self, value):
                    counts['spectrogram_exchanges'] += 1
                    return original_write.write(value)
                def flush(self): return original_write.flush()
                def close(self): return original_write.close()
            session.process.stdin = Input()
            report = dict(platform=sys.platform, scope='Windows source scientific session and recording service')
            report['spectrogram'] = measure(lambda: session.render(wav.getvalue(), **view))
            assert report['spectrogram']['sha256'] == digest(initial)
            report['recording'] = measure(lambda: service.dispatch(dict(op='preview', id='synthetic', width=900, spectrum=True, channel=0)))
            report['counts'] = counts
            if args.expect_optimized:
                assert counts == dict(recording_scans=1, recording_spectra=1, spectrogram_exchanges=0), counts
            report['success'] = True
            (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(dict(output=str(out), **report)))
        finally:
            recording.iter_audio, recording.DisplaySpectrum = original_iter, original_display
            session.close(); service.close()


if __name__ == '__main__': main()
