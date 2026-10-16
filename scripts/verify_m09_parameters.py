"""Compare built-in float32 inference against the original frozen Torch net."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ptb_worker.m09_parameters import predict, weights
from ptb_worker.m09_neural_models import ParameterRegressor, physical
from verification_artifacts import verification_run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = ParameterRegressor().eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    reference.load_state_dict(checkpoint['model'], strict=True)
    for key, value in weights().items():
        np.testing.assert_array_equal(value, checkpoint['model'][key].numpy())
    rng = np.random.default_rng(20261015)
    images = [('black', np.zeros((192, 256), np.uint8)), ('white', np.full((80, 100), 255, np.uint8)),
              ('ramp', np.tile(np.arange(256, dtype=np.uint8), (192, 1))),
              ('noise', rng.integers(0, 256, (95, 373), dtype=np.uint8))]
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root/'temporary-site/assets/spectrogram-examples').glob('*-input.png')):
        with Image.open(path) as image:images.append((path.name, np.asarray(image.convert('L'))))
    with verification_run('m09-parameters', 'numpy', 'fixed float32 weights vs original CPU Torch; linear/log metadata and original images') as (out, scratch):
        rows = []
        for name, gray in images:
            for seconds, lower, upper, axis in [(.2, 0, 8000, 'linear'), (1., 0, 11025, 'linear'),
                                                 (2., 50, 8000, 'log'), (30., 100, 48000, 'log')]:
                config = dict(time_start=0, time_end=seconds, freq_start=lower, freq_end=upper, frequency_axis=axis)
                resized = np.asarray(Image.fromarray(gray).resize((256,192), Image.Resampling.BILINEAR)).copy()
                meta = torch.tensor([[math.log2(seconds), math.log2(upper/5000), lower/1000, float(axis=='log')]])
                with torch.inference_mode():
                    expected = physical(reference(torch.from_numpy(resized[None,None]).float()/255, meta))[0].numpy()
                # Bounds errors must match the original physical output too.
                valid = bool(np.all((expected > 0) & (expected <= 160)))
                start = time.perf_counter()
                if not valid:
                    try:predict(gray, config)
                    except ValueError as exc:assert str(exc)=='m09_prediction_out_of_range'
                    else:raise AssertionError('Expected range rejection')
                    rows.append(dict(image=name,config=config,range_rejected=True));continue
                actual = predict(gray, config)
                elapsed = time.perf_counter() - start
                values = [actual['predicted_window_ms'], actual['dynamic_range_db']]
                # Float32 accumulation order differs; tolerance <0.002% and
                # far below a display unit. Integer windows must match exactly.
                np.testing.assert_allclose(values, expected, rtol=2e-5, atol=2e-5)
                assert actual['win_length_ms'] == max(1, math.floor(float(expected[0])+.5))
                rows.append(dict(image=name,config=config,reference=expected.tolist(),actual=values,
                                 max_absolute_error=float(np.max(np.abs(expected-values))),seconds=elapsed))
        write_json(out/'report.json', dict(success=True, cases=rows,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            max_absolute_error=max(r.get('max_absolute_error', 0) for r in rows),
            median_seconds=float(np.median([r['seconds'] for r in rows if 'seconds' in r]))))
        print('REPORT='+str(out/'report.json'), flush=True)


if __name__ == '__main__':main()
