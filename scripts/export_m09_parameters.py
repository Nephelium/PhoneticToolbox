"""Export weights only from a trusted local checkpoint to built-in NumPy data.

Run with the existing CPU research environment. Torch is only used at export;
the application reads arrays with allow_pickle=False. No training data/config
or optimizer state is included.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    raw = args.checkpoint.read_bytes()
    checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    arrays = {key: value.numpy() for key, value in checkpoint['model'].items()}
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Expected finite float32 inference weights')
    target = Path(__file__).resolve().parents[1] / 'packages/phonetic_core/src/phonetic_core/spec2wav/data'
    target.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    provenance = dict(schema='ptb-spec2wav-parameters/1', model='parameter-regressor-stream-v5',
        checkpoint_sha256=hashlib.sha256(raw).hexdigest(), epoch=checkpoint.get('epoch'), step=checkpoint.get('step'),
        parameters=sum(v.size for v in arrays.values()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
        dtype='float32', input=[1, 1, 192, 256], outputs=['window_ms', 'dynamic_range_db'],
        preprocessing='Pillow grayscale bilinear 256x192; /255; log2(seconds),log2(fmax/5000),fmin/1000,is_log',
        origin='Project-trained model; training data and optimizer are excluded.',
        distribution='Local integration; public redistribution conditions remain under review.')
    (target / 'parameters-v5.npz').write_bytes(data)
    (target / 'parameters-v5.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == '__main__':
    main()
