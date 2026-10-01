"""Read-only natural LPC regression; run through Invoke-M03-Python.ps1."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from uuid import uuid4

from scipy.io import wavfile
from ptb_worker.lpc_child import prepare
from ptb_worker.segmentation import unpack_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--roi', type=float, nargs=2, action='append', required=True)
    args = parser.parse_args()
    source = args.audio.resolve(strict=True)
    grid_file = source.with_suffix('.TextGrid')
    raw, grid = source.read_bytes(), grid_file.read_bytes()
    hashes = [hashlib.sha256(b).hexdigest() for b in (raw, grid)]
    out = Path(__file__).resolve().parents[1] / 'output/validation/m04-r1' / uuid4().hex
    out.mkdir(parents=True)
    report = dict(source=str(source), input_sha256=hashes[0], textgrid_sha256=hashes[1], cases=[])

    def files(config, annotation):
        bundle = unpack_bundle(prepare(raw, config, 'natural.wav', annotation), 8_000_000)
        return dict(zip([entry['name'] for entry in bundle.manifest['files']], bundle.payloads))

    for index, (start, end) in enumerate(args.roi):
        config = dict(roi_start=start, roi_end=end)
        labelled, plain = files(config, grid), files(config, None)
        metadata = json.loads(labelled['lpc.ptb.json'])
        baseline = json.loads(plain['lpc.ptb.json'])
        assert metadata['spectrum'] == baseline['spectrum']
        assert metadata['selection'] == baseline['selection']
        assert labelled['lpc_AUDIO.wav'] == plain['lpc_AUDIO.wav']
        rate, audio = wavfile.read(io.BytesIO(labelled['lpc_AUDIO.wav']))
        assert len(audio) == int(end * rate) - int(start * rate)
        folder = out / str(index + 1)
        folder.mkdir()
        for name, payload in labelled.items():
            (folder / name).write_bytes(payload)
        report['cases'].append(dict(roi=metadata['selection'], label=metadata['label'],
                                   spectrum_points=len(metadata['spectrum']['magnitude_db']),
                                   same_spectrum_and_wav_without_textgrid=True))
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, grid_file)]
    report.update(success=True, original_hashes_unchanged=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out / 'report.json')


if __name__ == '__main__':
    main()
