"""Built-in offline parameter prediction; no optional component or Torch."""
from functools import lru_cache
import hashlib
import io
import json
import math
from importlib.resources import files

import numpy as np
from PIL import Image
from phonetic_core.spec2wav.parameters import regress


@lru_cache(maxsize=1)
def weights():
    resource = files('phonetic_core.spec2wav').joinpath('data')
    try:
        provenance = json.loads(resource.joinpath('parameters-v5.json').read_text('utf-8'))
        raw = resource.joinpath('parameters-v5.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest() != provenance['sha256']:
            raise ValueError('m09_parameter_model_invalid')
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            result = {key: archive[key] for key in archive.files}
        for value in result.values():
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('m09_parameter_model_invalid')
            value.flags.writeable = False
        return result
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError('m09_parameter_model_invalid') from exc


def predict(gray, config):
    seconds = config['time_end'] - config['time_start']
    lower, upper = config['freq_start'], config['freq_end']
    axis = config.get('frequency_axis', 'linear')
    if not all(math.isfinite(v) for v in (seconds, lower, upper)) or seconds <= 0 or not 0 <= lower < upper or axis not in ('linear', 'log') or (axis == 'log' and lower <= 0):
        raise ValueError('invalid_spectrogram_config')
    resized = np.asarray(Image.fromarray(gray).resize((256, 192), Image.Resampling.BILINEAR))
    meta = [math.log2(seconds), math.log2(upper / 5000), lower / 1000, float(axis == 'log')]
    values = regress(resized, meta, weights()).tolist()
    if not np.isfinite(values).all() or not 0 < values[0] <= 160 or not 0 < values[1] <= 160:
        raise ValueError('m09_prediction_out_of_range')
    ms = max(1, math.floor(values[0] + .5))
    return dict(predicted_window_ms=values[0], win_length_ms=ms, hop_ms=ms / 4,
                dynamic_range_db=values[1], min_db=-values[1], max_db=0,
                phase_initialization='pulse' if ms < 15 else 'random', absolute_db_predicted=False)
