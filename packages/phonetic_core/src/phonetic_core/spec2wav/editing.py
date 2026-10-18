"""M09-R1 array-only geometry, brush compositing and original-phase editing.

NumPy/SciPy/OpenCV dependencies are registered in the project source registry.
Image legacy reconstruction remains in reconstruction.py.
"""
import numpy as np
from cv2 import LINE_8, circle, getPerspectiveTransform, line, warpPerspective
from scipy.signal import stft, istft


def rectify(gray, corners=None):
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 2 or gray.size > 25_000_000:
        raise ValueError('invalid_spectrogram_image')
    if corners:
        if len(corners) != 4:
            raise ValueError('invalid_image_corners')
        points = np.float32([[p['x']*(gray.shape[1]-1), p['y']*(gray.shape[0]-1)] for p in corners])
        edges = np.roll(points, -1, axis=0)-points
        cross = edges[:, 0]*np.roll(edges, -1, axis=0)[:, 1]-edges[:, 1]*np.roll(edges, -1, axis=0)[:, 0]
        if not np.all(np.isfinite(points)) or not np.all(cross > 1):
            raise ValueError('invalid_image_corners')
        width = max(int(np.linalg.norm(points[1]-points[0])), int(np.linalg.norm(points[2]-points[3])))
        height = max(int(np.linalg.norm(points[3]-points[0])), int(np.linalg.norm(points[2]-points[1])))
        if min(width, height) < 2 or width*height > 1_000_000:
            raise ValueError('invalid_image_corners')
        target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        gray = warpPerspective(gray, getPerspectiveTransform(points, target), (width, height))
    if gray.size > 1_000_000:
        raise ValueError('spectrogram_budget')
    return gray.copy()


def pixel_brush(mask,points,diameter):
    """Binary round caps on pixel centers; 1 px is exactly one pixel."""
    radius=diameter/2
    centers=points.astype(float)+(0.5 if diameter%2 else 0.)
    segments=list(zip(centers[:-1],centers[1:])) if len(centers)>1 else [(centers[0],centers[0])]
    for a,b in segments:
        low=np.maximum(0,np.floor(np.minimum(a,b)-radius).astype(int))
        high=np.minimum([mask.shape[1],mask.shape[0]],np.ceil(np.maximum(a,b)+radius).astype(int))
        if np.any(high<=low):continue
        yy,xx=np.ogrid[low[1]:high[1],low[0]:high[0]]
        dx,dy=b-a;denominator=dx*dx+dy*dy
        t=np.clip(((xx+.5-a[0])*dx+(yy+.5-a[1])*dy)/denominator,0,1) if denominator else 0
        hit=(xx+.5-a[0]-t*dx)**2+(yy+.5-a[1]-t*dy)**2<=radius**2
        mask[low[1]:high[1],low[0]:high[0]] |= hit.astype(np.uint8)


def paint(gray, strokes):
    """One alpha composite per stroke, regardless of pointer sampling density."""
    result = np.asarray(gray, dtype=np.float64).copy()
    height, width = result.shape
    touched = np.zeros(result.shape, bool)
    if len(strokes) > 256 or sum(len(s['points']) for s in strokes) > 8192:
        raise ValueError('spectrogram_budget')
    for stroke in strokes:
        alpha = stroke['opacity']
        if alpha == 0:
            continue
        canvas_width = width if stroke.get('canvas_width') is None else stroke['canvas_width']
        canvas_height = height if stroke.get('canvas_height') is None else stroke['canvas_height']
        fraction = 1. if stroke.get('frequency_fraction') is None else stroke['frequency_fraction']
        if (not 2 <= canvas_width <= 32768 or not 2 <= canvas_height <= 32768
                or canvas_width*canvas_height > 6_000_000 or not 0 < fraction <= 1):
            raise ValueError('spectrogram_budget')
        mask = np.zeros((canvas_height, canvas_width), np.uint8)
        points = np.rint([[p['x']*(canvas_width-1), p['y']*(canvas_height-1)] for p in stroke['points']]).astype(np.int32)
        radius = max(1, int(stroke['size']/2+.5))
        if stroke.get('size_unit')=='px':
            pixel_brush(mask,points,stroke['size'])
        # LINE_8 matches deterministic pixel grid; round caps include single dots.
        else:
            for a, b in zip(points[:-1], points[1:]):
                line(mask, tuple(a), tuple(b), 1, radius*2, LINE_8)
            for p in (points[0], points[-1]):
                circle(mask, tuple(p), radius, 1, -1, LINE_8)
        if canvas_width != width:
            # Average covered drawing pixels in each original time cell. The
            # STFT grid/phase stay unchanged; unpainted cells retain their value.
            edges = np.linspace(0, canvas_width, width+1)
            indices = np.floor(edges).astype(int)
            prefix = np.pad(np.cumsum(mask, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
            integral = prefix[:, indices] + mask[:, np.minimum(indices, canvas_width-1)]*(edges-indices)
            coverage = np.diff(integral, axis=1)/(canvas_width/width)
        else:
            coverage = mask
        if canvas_height != height or fraction != 1:
            # The visible band occupies the lower fraction of the full image.
            # Integrate clipped row coverage, leaving all other bins untouched.
            step = canvas_height/(height*fraction)
            edges = np.clip((np.arange(height+1)-height*(1-fraction))*step, 0, canvas_height)
            indices = np.floor(edges).astype(int)
            prefix = np.pad(np.cumsum(coverage, axis=0, dtype=np.float64), ((1, 0), (0, 0)))
            integral = prefix[indices] + coverage[np.minimum(indices, canvas_height-1)]*(edges-indices)[:, None]
            coverage = np.clip(np.diff(integral, axis=0)/step, 0, 1)
        hit = coverage > 0
        amount = alpha*coverage[hit]
        result[hit] = result[hit]*(1-amount)+stroke['color']*amount
        touched |= hit
    return result, touched


def analyze_audio(audio, sr, *, channel=0, n_fft=1024, dynamic_range=60):
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    if (samples.ndim != 2 or not 1 <= samples.shape[1] <= 2 or not 8000 <= sr <= 96000
            or not 2 <= len(samples) <= 30*sr or not 0 <= channel < samples.shape[1]
            or not np.isfinite(samples).all() or n_fft not in (512, 1024, 2048)
            or not 20 <= dynamic_range <= 120):
        raise ValueError('invalid_spectrogram_audio')
    hop = n_fft//4
    if (n_fft//2+1)*(2+int(np.ceil(len(samples)/hop))) > 6_000_000:
        raise ValueError('spectrogram_budget')
    selected = np.pad(samples[:, channel], (0, max(0, n_fft-len(samples))))
    _, _, spectrum = stft(selected, sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop,
                          nfft=n_fft, boundary='zeros', padded=True)
    magnitude = np.abs(spectrum)
    reference = float(magnitude.max()) or .05
    gray = np.flipud(np.clip(-20*np.log10(np.maximum(magnitude, reference*1e-12)/reference)/dynamic_range*255, 0, 255))
    metadata = dict(sample_rate=sr, samples=len(samples), channels=samples.shape[1], channel=channel,
                    duration=len(samples)/sr, n_fft=n_fft, hop_length=hop, dynamic_range=dynamic_range,
                    reference_amplitude=reference, width=gray.shape[1], height=gray.shape[0])
    return samples, spectrum, gray, metadata


def edit_audio(audio, sr, strokes, *, channel=0, n_fft=1024, dynamic_range=60):
    samples, original, gray, metadata = analyze_audio(audio, sr, channel=channel, n_fft=n_fft, dynamic_range=dynamic_range)
    painted, touched = paint(gray, strokes)
    modified = original.copy()
    target_magnitude = metadata['reference_amplitude']*10**(-np.flipud(painted)/255*dynamic_range/20)
    hit = np.flipud(touched)
    modified[hit] = target_magnitude[hit]*np.exp(1j*np.angle(original[hit]))
    _, reconstructed = istft(modified, sr, window='hann', nperseg=n_fft, noverlap=n_fft-n_fft//4,
                              nfft=n_fft, input_onesided=True, boundary=True)
    reconstructed = reconstructed[:len(samples)]
    gain = 1.
    if touched.any() and np.max(np.abs(reconstructed)) > 1:
        gain = .99/float(np.max(np.abs(reconstructed)))
        reconstructed *= gain
    output = samples.copy()
    output[:, channel] = reconstructed
    # Same reference and range on both images, including any output attenuation.
    _, comparison, _, _ = analyze_audio(output, sr, channel=channel, n_fft=n_fft, dynamic_range=dynamic_range)
    comparison_gray = np.flipud(np.clip(-20*np.log10(np.maximum(np.abs(comparison), metadata['reference_amplitude']*1e-12)/metadata['reference_amplitude'])/dynamic_range*255, 0, 255))
    metadata.update(schema_version='m09-audio-draw/1', phase_method='original-stft-phase',
                    zero_magnitude_phase='zero-radians', changed_bins=int(touched.sum()), output_gain=gain,
                    magnitude_mapping='20log10-amplitude', n_iter=0, wav_subtype='FLOAT',
                    warning='Modified STFT is projected by inverse STFT; reanalysis can differ from the drawn target.')
    return dict(audio=output, sr=sr, target=np.rint(painted).astype(np.uint8),
                image=np.rint(comparison_gray).astype(np.uint8), metadata=metadata)
