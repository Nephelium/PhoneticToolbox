"""Read-only Linux interoperability checks; no Linux GUI or scientific job claim."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'output/m03-r4/wsl-manifest.json').read_text('utf-8'))
checks = []
for relative, expected in manifest['files'].items():
    raw = (root / relative).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    raw.decode('utf-8')
    checks.append('same UTF-8 built resource: ' + relative)
metadata = json.loads((root / manifest['metadata']).read_text('utf-8'))
assert metadata['config']['highpass_cutoff'] == 25
assert metadata['config']['lowpass_cutoff'] == 1500
assert metadata['config']['roi_start'] == 0 and metadata['config']['roi_end'] is None
checks.append('Windows batch metadata readable with 25/1500 Hz and whole-file ROI')
report = {'success': True, 'platform': 'WSL NInfer', 'checks': checks,
          'limits': 'Static resource and saved metadata reading only; Linux GUI/playback/jobs not tested.'}
(root / 'output/m03-r4/wsl-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
