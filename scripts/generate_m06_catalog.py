"""Deterministic UI catalog from pure core constants; --check detects drift."""
import json,sys
from pathlib import Path
from phonetic_core.synthesis.klatt.api import defaults
from phonetic_core.synthesis.klatt.klatt_config import PARAM_DEFAULTS
from phonetic_core.synthesis.klatt.input_parser import VOWEL_FORMANTS
root=Path(__file__).resolve().parents[1]
data=dict(defaults=defaults(),parameters=PARAM_DEFAULTS,vowels=VOWEL_FORMANTS,presets=json.loads((root/'packages/phonetic_core/src/phonetic_core/synthesis/klatt/presets.json').read_text('utf8')))
target=root/'frontend/src/modules/speech-synthesis/catalog.json'
text=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
if '--check' in sys.argv:
 assert json.loads(target.read_text('utf8'))==json.loads(text),'M06 UI catalog drift'
else:target.write_text(text,encoding='utf8')
print(f'M06 catalog: {len(PARAM_DEFAULTS)} parameters, defaults, vowels and five presets agree')
