"""M08/1: serial framed outputs from the existing scientific handler.

The only writable file is a host-preallocated, quota-reserved native WAV slot.
No batch audio accumulation, user paths, database or unbounded output directory.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys

MAX_BYTES = 64_000_000


def run():
    # Per-child thread budget, set before loading NumPy/SciPy. No host/system change.
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[variable]='1'
    from .m08_jobs import execute, read_authorized
    from phonetic_core.manipulation.m08_rules import track
    from .io.scratch import no_links
    import parselmouth
    from scipy.io import wavfile
    with open(sys.argv[2], 'wb', buffering=0) as stream:
        total = 0
        def frame(header, raw=b''):
            nonlocal total
            encoded = json.dumps(header, ensure_ascii=False, allow_nan=False).encode()
            total += 4 + len(encoded) + len(raw)
            if total > MAX_BYTES or len(encoded) > 2_000_000:
                raise ValueError('m08_output_budget')
            stream.write(struct.pack('<I', len(encoded))); stream.write(encoded)
            for offset in range(0, len(raw), 65536):
                stream.write(raw[offset:offset+65536])
        def emit_file(name, raw):
            frame(dict(kind='file', name=name, size=len(raw), sha256=hashlib.sha256(raw).hexdigest()), raw)
        try:
            header = json.loads(Path(sys.argv[1]).read_text('utf-8'))
            sound = read_authorized(header['audio_path'], header['sha256'])
            native = Path(header['native_path']); no_links(native)
            if native.stat().st_size or native.parent != Path(sys.argv[1]).parent:
                raise ValueError('m08_native_scratch')
            results = []
            def emit(name, output, snapshot):
                no_links(native)
                if output.n_samples*output.n_channels*2+4096>MAX_BYTES:
                    raise ValueError('m08_output_budget')
                output.save(str(native), 'WAV')  # Exact V2/Praat PCM16 encoding.
                if not 0 < native.stat().st_size <= MAX_BYTES:
                    raise ValueError('m08_output_budget')
                saved = parselmouth.Sound(str(native))
                times, f0 = track(saved)
                raw = native.read_bytes()
                emit_file(name, raw)
                item = dict(name=name, start=snapshot['source_start_s'], end=snapshot['source_end_s'],
                            times=times.tolist(), original_f0=f0.tolist(), config=snapshot['config'])
                results.append(item)
                with native.open('r+b') as file: file.truncate(0)
                return name
            info = execute(sound, header['config'], Path(header['input_name']).stem, emit=emit)
            if header['config']['action'] == 'preview':
                if sound.n_samples*sound.n_channels*8+4096>MAX_BYTES:
                    raise ValueError('m08_output_budget')
                wav = io.BytesIO()
                wavfile.write(wav, int(sound.sampling_frequency), sound.values.T.copy())
                emit_file('m08-preview.wav', wav.getvalue())
            metadata = dict(info, audio_sha256=header['sha256'], results=results)
            emit_file('m08.ptb.json', json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode())
            frame(dict(kind='complete', audio_sha256=header['sha256'], count=len(results)+1+(header['config']['action']=='preview')))
        except Exception as exc:
            code = str(exc)
            frame(dict(kind='error', code=code if code.startswith('m08_') and len(code)<80 else 'm08_execution_failed'))


if __name__ == '__main__':
    run()
