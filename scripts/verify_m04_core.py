"""Installed M04 wheel path, numerical payload and runtime evidence."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--installed-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import scipy
    import phonetic_core.lpc as lpc
    import phonetic_core.lpc._legacy as legacy
    root = Path(__file__).resolve().parents[1]
    package = Path(lpc.__file__).resolve()
    assert package.is_relative_to(args.installed_root.resolve()), package
    assert not package.is_relative_to(root/'packages'), package
    assert not any(x.split('.')[0] in {'PyQt6', 'PySide6', 'fastapi', 'parselmouth'} for x in sys.modules)
    assert (package.parent/'NOTICE.txt').is_file()
    assert np.__version__ == '2.2.6' and scipy.__version__ == '1.16.3'
    # The numerical payload equals the V2 snapshot in the initial migration
    # except its one-line module docstring (the public wrapper is separate).
    source = root.parent/'PhoneticToolbox_v2'
    manifest = json.loads((root/'tests/fixtures/m04/manifest.json').read_text('utf-8'))
    for relative, expected in manifest['source_hashes'].items():
        assert hashlib.sha256((source/relative).read_bytes()).hexdigest() == expected, relative
    actual = Path(legacy.__file__).read_text('utf-8').split('\n', 1)[1]
    assert actual == (source/'phonetic_toolbox/core/acoustic/lpc.py').read_text('utf-8')
    # Exercise the published maximum through the validated public API.
    before = np.random.default_rng(404).normal(size=lpc.MAX_ROI_SAMPLES)
    sample_hash = hashlib.sha256(before.tobytes()).hexdigest()
    started = time.perf_counter()
    result = lpc.compute_spectrum(before, 48000, lpc.LPCConfig(order=200))
    elapsed = time.perf_counter()-started
    assert hashlib.sha256(before.tobytes()).hexdigest() == sample_hash
    assert np.isfinite(result.magnitude_db).all() and len(result.magnitude_db) == 1024
    report = dict(module_path=str(package), python=sys.executable,
                  core_version=importlib.metadata.version('phonetic-core'),
                  numpy=np.__version__, scipy=scipy.__version__,
                  legacy_payload_equal=True, source_files_unchanged=len(manifest['source_hashes']),
                  max_roi_samples=lpc.MAX_ROI_SAMPLES, max_roi_seconds_48khz=1,
                  max_order=200, max_roi_elapsed=elapsed, points=1024,
                  scope='Windows installed core target with existing Conda/MKL scientific runtime; not a clean full environment')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
