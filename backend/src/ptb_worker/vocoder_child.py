"""Offline BigVGAN adapter, executed with the optional component's Python.

This source is frozen into the component by build_m09_vocoder_component.py.
The application has no Torch import or dependency when the option is off.
"""
import io
import json
import sys
from pathlib import Path


def devices():
    import torch
    available = torch.cuda.is_available()
    if available:
        try:
            # A detected card can still lack compatible kernels/driver support.
            value = torch.ones((2,2),device='cuda')
            assert float((value @ value).sum().cpu()) == 8
        except Exception:
            available = False
    return dict(cpu=True,cuda=available,cuda_runtime=bool(torch.version.cuda),
                gpu_name=torch.cuda.get_device_name(0) if available else None)


def select_device(requested):
    if requested not in ('auto','cpu','cuda'):raise ValueError('invalid_spectrogram_config')
    if requested=='cpu':return 'cpu'
    available=devices()['cuda']
    if requested=='cuda' and not available:raise ValueError('m09_gpu_unavailable')
    return 'cuda' if available else 'cpu'


def synthesize(audio, sr, model_root, feature_transform=None, device='auto'):
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from math import gcd
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not 2 <= len(audio) <= sr * 30 or not 2 <= sr <= 96000 or not np.isfinite(audio).all():
        raise ValueError('m09_vocoder_input_invalid')
    torch.set_num_threads(4)
    device=select_device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0, str(model_root))
    from bigvgan import BigVGAN
    from env import AttrDict
    from meldataset import get_mel_spectrogram
    config = AttrDict(json.loads((model_root / 'config.json').read_text('utf-8')))
    model = BigVGAN(config, use_cuda_kernel=False)
    # Upstream remove_weight_norm prints to stdout, reserved for binary output.
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        model.load_state_dict(torch.load(model_root / 'bigvgan_generator.pt', map_location='cpu', weights_only=True)['generator'], strict=True)
        model.remove_weight_norm()
    model.eval().to(device)
    input_gain = min(1., .65 / max(float(np.max(np.abs(audio))), 1e-12))
    common = gcd(sr, config.sampling_rate)
    native = resample_poly(audio * input_gain, config.sampling_rate // common, sr // common)
    length = len(native)
    # Provide the final partial frame, then trim. Preserve duration, never stretch.
    pad = max(config.n_fft, ((length+config.hop_size-1)//config.hop_size)*config.hop_size) - length
    native = np.pad(native, (0, pad), mode='reflect')
    with torch.inference_mode():
        features = get_mel_spectrogram(torch.tensor(native[None], dtype=torch.float32,device=device), config)
        if feature_transform is not None:
            features = feature_transform(features, config)
        output = model(features).reshape(-1).cpu().numpy()[:length]
    restored = resample_poly(output.astype(float), sr // common, config.sampling_rate // common)
    if len(restored) < len(audio):
        raise ValueError('m09_vocoder_output_invalid')
    restored = restored[:len(audio)]
    if not np.isfinite(restored).all():
        raise ValueError('m09_vocoder_output_invalid')
    gain = min(1., .95 / max(float(np.max(np.abs(restored))), 1e-12))
    return restored * gain, dict(model='BigVGAN base 22 kHz',
        model_revision='760f0c694aba68b8c200dd52ffd3b4ecafa7028c',
        inference_sample_rate=config.sampling_rate, inference_device=device, input_gain=input_gain,
        output_gain=gain, native_samples=length, sample_rate=sr, samples=len(audio),
        time_alignment='pad final inference frame, trim to input duration; no time stretch')


def main():
    import numpy as np
    if sys.argv[1:]==['--devices']:
        print(json.dumps(devices()));return
    model_root = Path(sys.argv[2]) if len(sys.argv)==3 and sys.argv[1]=='--model-root' else Path(__file__).resolve().parent / 'model'
    if sys.argv[1:] == ['--check']:
        t = np.arange(16000) / 16000
        source = .1 * np.sin(2 * np.pi * 150 * t) + .03 * np.sin(2 * np.pi * 450 * t)
        output, meta = synthesize(source, 16000, model_root)
        if len(output) != len(source) or np.std(output) < 1e-6:
            raise ValueError('m09_vocoder_self_test_failed')
        print(json.dumps(dict(success=True, samples=len(output), **{'metadata':meta})))
        return
    raw = sys.stdin.buffer.read(32_000_001)
    if not 0 < len(raw) <= 32_000_000:
        raise ValueError('m09_vocoder_input_invalid')
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        audio, sr = data['audio'], int(data['sr'])
        device=str(data['device']) if 'device' in data else 'auto'
    output, meta = synthesize(audio, sr, model_root,device=device)
    stream = io.BytesIO()
    np.savez(stream, audio=output, metadata=json.dumps(meta))
    sys.stdout.buffer.write(stream.getvalue())


if __name__ == '__main__':
    main()
