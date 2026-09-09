import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from phonetic_core.models.audio import AudioInput
from phonetic_core.acoustic.reaper_codec import reaper_pcm16

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('conversion_recipes', ROOT/'scripts/capture_m01_conversion_baseline.py')
# Its baseline-support import is a script helper, not a production dependency.
import sys
support_spec = importlib.util.spec_from_file_location('baseline_support', ROOT/'scripts/baseline_support.py')
support = importlib.util.module_from_spec(support_spec)
support_spec.loader.exec_module(support)
sys.modules.setdefault('baseline_support', support)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)
GOLDEN = json.loads((ROOT/'tests/fixtures/m01-conversion.json').read_text('utf-8'))


@pytest.mark.parametrize('case', GOLDEN['cases'], ids=lambda c: c['filename'])
def test_exact_reaper_conversion_against_original_function(case, tmp_path):
    assert GOLDEN['producer'] == 'original-v2-conversion-only'
    data = recipe.samples(case['rate'], case['dtype'], case['channels'])
    source = tmp_path/'original.wav'
    wavfile.write(source, case['rate'], data)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case['input_sha256']
    pcm = reaper_pcm16(AudioInput(data, case['rate']))
    assert pcm.dtype == np.int16 and len(pcm) == case['output_frames']
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == case['pcm_sha256']
    destination = tmp_path/'converted.wav'
    wavfile.write(destination, 16000, pcm)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == case['wav_sha256']
