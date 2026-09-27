"""M15 Linux static-artifact audit only. No server, database or audio claim."""
from __future__ import annotations
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
dist = root / 'frontend/dist'
assert sys.platform == 'linux', 'Run with the existing native WSL interpreter'
assert (dist / 'index.html').is_file()
targets = [dist / 'index.html']
for pattern in ('AppShell-*.js', 'PerceptionPage-*.js', 'PerceptionPage-*.css', 'DoulosSIL-Regular-*.ttf'):
    matches = list((dist / 'assets').glob(pattern))
    assert len(matches) == 1, (pattern, len(matches))
    targets.extend(matches)
html = (dist / 'index.html').read_text('utf8')
for relative in re.findall(r'(?:src|href)="(/assets/[^"]+)"', html):
    assert (dist / relative.lstrip('/')).is_file(), relative
targets.extend(sorted((dist / 'assets').glob('index-*.js')))
targets.extend(sorted((dist / 'assets').glob('index-*.css')))
files = [{'path': str(p.relative_to(dist)), 'bytes': p.stat().st_size,
          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in targets]
report = {'success': True, 'platform': platform.platform(), 'python': sys.version,
          'scope': 'native Linux filesystem, complete entry resources and M15 bundle byte audit',
          'linux_browser_playback_verified': False, 'network_requests': 0,
          'server_compute': 'not applicable', 'files': files}
out = root / 'output/validation/m15-linux-static' / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
out.mkdir(parents=True)
(out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(out / 'report.json')
