"""Generate settings-only presentation data from the current native engine.

No audio, user settings, computed geometry or scientific results are included.
"""
import hashlib
import json
from pathlib import Path
from workbench_source import bind_sources

ROOT = Path(__file__).resolve().parents[1]
bind_sources(ROOT)
from phonetic_core.vocal_tract.engine import Engine

engine = Engine(resource_dir=ROOT / 'resources/vocal_tract/native')
try:
    data = engine.metadata()
finally:
    engine.close()
data['presentation_source'] = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (ROOT / 'resources/vocal_tract/native').iterdir()
    if p.name in ('geometry_p2.dll', 'VocalTractLabApi.dll', 'VocalTractLabAnalysis.dll', 'JD2.speaker')
}
(ROOT / 'frontend/public/vocal-tract/presentation-metadata.json').write_text(
    json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
