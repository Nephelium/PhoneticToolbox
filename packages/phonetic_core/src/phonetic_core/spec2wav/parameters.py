"""Float32 NumPy inference for the frozen window/dynamic-range regressor.

The four stride-2 convolutions, GroupNorm(4, eps=1e-5), SiLU,
adaptive average pool and two linear layers match ParameterRegressor.eval().
Image decoding/resizing and model resource I/O belong to the worker adapter.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _silu(x):
    return x / (np.float32(1) + np.exp(-x))


def regress(gray, metadata, weights):
    """Return physical [window_ms, dynamic_range_db] without changing inputs."""
    gray = np.asarray(gray)
    metadata = np.asarray(metadata, dtype=np.float32)
    if gray.shape != (192, 256) or gray.dtype != np.uint8 or metadata.shape != (4,) or not np.isfinite(metadata).all():
        raise ValueError('invalid_spectrogram_config')
    x = gray[None].astype(np.float32) / np.float32(255)
    for layer in (0, 3, 6, 9):
        w = weights[f'encoder.{layer}.weight']
        patches = sliding_window_view(np.pad(x, ((0, 0), (1, 1), (1, 1))), (3, 3), axis=(1, 2))[:, ::2, ::2]
        height, width = patches.shape[1:3]
        # Each row uses the original Conv2d C, kernel-y, kernel-x order.
        columns = patches.transpose(1, 2, 0, 3, 4).reshape(height * width, -1)
        x = (columns @ w.reshape(w.shape[0], -1).T + weights[f'encoder.{layer}.bias']).T.reshape(-1, height, width)
        grouped = x.reshape(4, -1)
        mean = grouped.mean(axis=1, keepdims=True)
        centered = grouped - mean
        normalized = centered / np.sqrt((centered * centered).mean(axis=1, keepdims=True) + np.float32(1e-5))
        x = normalized.reshape(x.shape) * weights[f'encoder.{layer+1}.weight'][:, None, None] + weights[f'encoder.{layer+1}.bias'][:, None, None]
        x = _silu(x)
    # The fixed 192x256 input gives 12x16 features: each pool cell is 4x4.
    x = x.reshape(48, 3, 4, 4, 4).mean(axis=(2, 4)).reshape(-1)
    x = np.concatenate((x, metadata))
    x = _silu(weights['head.0.weight'] @ x + weights['head.0.bias'])
    z = weights['head.3.weight'] @ x + weights['head.3.bias']
    return np.exp2(z) * np.array([10, 60], dtype=np.float32)
