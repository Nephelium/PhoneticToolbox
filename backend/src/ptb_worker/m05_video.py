"""M05 local/offline adapter: lazy legacy inference, streaming PTS and bounded results.

This adapter accepts only host-authorized paths. It does not expose HTTP paths or
bypass the public job writer: cloud wiring must supply its own bounded sink.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time


@dataclass(frozen=True)
class VideoLimits:
    input_bytes: int = 1_000_000_000
    output_bytes: int = 512_000_000
    max_pixels: int = 1920 * 1080
    max_frames: int = 300_000
    timeout_seconds: float = 1800.


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


class JsonLinesSink:
    def __init__(self, stream, byte_limit):
        self.stream, self.limit, self.written = stream, byte_limit, 0

    def write(self, row):
        value = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
        if self.written + len(value) > self.limit: raise ValueError('output_budget_exceeded')
        count = self.stream.write(value)
        if count != len(value): raise OSError('short_result_write')
        self.written += count


def analyze_video(path, sink, config, *, limits=VideoLimits(), stop=lambda: False, progress=lambda value: None):
    """Every decoded frame is submitted; no sampling or frame_index/fps fallback."""
    import av
    import numpy as np
    import mediapipe as mp
    from phonetic_core.lip.sequence import LipSequence, mesh_neighbors
    import importlib.metadata as metadata
    path = Path(path)
    if not path.is_file() or path.stat().st_size > limits.input_bytes: raise ValueError('input_budget_exceeded')
    if mp.__version__ != '0.10.14': raise ValueError('legacy_runtime_version_mismatch')
    runtime_spec = Path(__file__).resolve().parents[3] / 'resources/m05/legacy-runtime.json'
    locked = json.loads(runtime_spec.read_text('utf-8'))
    if any(metadata.version(name) != version for name, version in locked['versions'].items()):
        raise ValueError('legacy_runtime_version_mismatch')
    for relative, expected in locked['models'].items():
        if sha256_file(Path(mp.__file__).parent / relative) != expected:
            raise ValueError('legacy_model_hash_mismatch')
    started = time.perf_counter()
    checks = lambda: stop() or time.perf_counter() - started > limits.timeout_seconds
    input_hash = sha256_file(path)
    if checks(): raise InterruptedError('cancelled_or_timeout')
    connections = set(mp.solutions.face_mesh.FACEMESH_TESSELATION) | set(mp.solutions.face_mesh.FACEMESH_CONTOURS)
    sequence = LipSequence(config, mesh_neighbors(connections))
    detected = 0
    decode_seconds = inference_seconds = 0.
    first_pts = last_pts = anchor = None
    dimensions = set()
    rotations = set()
    with av.open(str(path), mode='r') as container:
        if not container.streams.video: raise ValueError('no_video_stream')
        stream = container.streams.video[0]
        if stream.width * stream.height > limits.max_pixels: raise ValueError('resolution_budget_exceeded')
        audio = container.streams.audio[0] if container.streams.audio else None
        audio_start = float(audio.start_time * audio.time_base) if audio is not None and audio.start_time is not None else None
        audio_decoded_start = None
        if audio is not None:
            with av.open(str(path), mode='r') as audio_probe:
                first_audio = next(iter(audio_probe.decode(audio_probe.streams.audio[0])), None)
                if first_audio is not None:
                    if first_audio.pts is None or first_audio.time_base is None: raise ValueError('missing_audio_pts')
                    audio_decoded_start = float(first_audio.pts * first_audio.time_base)
        video_start = float(stream.start_time * stream.time_base) if stream.start_time is not None else None
        rotation = stream.metadata.get('rotate')
        nominal_rate=float(stream.average_rate) if stream.average_rate is not None else None
        with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                            min_detection_confidence=.5, min_tracking_confidence=.5) as mesh:
            iterator = iter(container.decode(stream))
            while True:
                if checks(): raise InterruptedError('cancelled_or_timeout')
                decode_start = time.perf_counter()
                try: frame = next(iterator)
                except StopIteration: break
                if sequence.index >= limits.max_frames: raise ValueError('frame_budget_exceeded')
                if frame.width * frame.height > limits.max_pixels: raise ValueError('resolution_budget_exceeded')
                if frame.pts is None or frame.time_base is None: raise ValueError('missing_decoded_pts')
                pts = float(frame.pts * frame.time_base)
                if not math.isfinite(pts) or (last_pts is not None and pts <= last_pts): raise ValueError('nonmonotonic_decoded_pts')
                if first_pts is None:
                    first_pts = pts
                    anchor = audio_decoded_start if audio_decoded_start is not None else pts
                image = frame.to_ndarray(format='rgb24')
                angle=float(frame.rotation)%360
                if angle not in (0,90,180,270):raise ValueError('unsupported_display_rotation')
                angle=int(angle)
                display_matrix=None
                for side in frame.side_data:
                    if side.type.name=='DISPLAYMATRIX':
                        display_matrix=np.frombuffer(bytes(side),dtype='<i4').tolist()
                        if len(display_matrix)!=9 or display_matrix[0]*display_matrix[4]-display_matrix[1]*display_matrix[3]<0:
                            raise ValueError('unsupported_display_reflection')
                # V2's OpenCV file decoder has ORIENTATION_AUTO=1. Preserve
                # that implicit input behavior before converting landmarks.
                if angle:image=np.ascontiguousarray(np.rot90(image,angle//90))
                effective_height,effective_width=image.shape[:2];rotations.add(angle)
                decode_seconds += time.perf_counter() - decode_start
                inference_start = time.perf_counter()
                result = mesh.process(image)
                inference_seconds += time.perf_counter() - inference_start
                points = None
                if result.multi_face_landmarks:
                    points = np.array([(p.x * effective_width, p.y * effective_height) for p in result.multi_face_landmarks[0].landmark], dtype=np.float32)
                    detected += 1
                row = sequence.process(points, pts - anchor)
                row.update(pts=frame.pts, time_base=[frame.time_base.numerator, frame.time_base.denominator],
                           media_time_s=pts, width=effective_width, height=effective_height,
                           coded_width=frame.width,coded_height=frame.height,display_rotation_degrees=angle,display_matrix=display_matrix)
                sink.write(row)
                dimensions.add((effective_width, effective_height))
                last_pts = pts
                if sequence.index % 10 == 0:
                    progress(dict(phase='analyzing', frames=sequence.index, detected=detected, media_time_s=pts))
    if not sequence.index: raise ValueError('empty_decoded_video')
    if checks(): raise InterruptedError('cancelled_or_timeout')
    # Reject concurrently changed inputs; no success manifest for a mixed source.
    if sha256_file(path) != input_hash: raise ValueError('input_changed_during_analysis')
    elapsed = time.perf_counter() - started
    model_root = Path(mp.__file__).parent / 'modules'
    models = {str(p.relative_to(model_root)): sha256_file(p) for p in model_root.rglob('*.tflite') if 'face_' in str(p)}
    return dict(schema='m05-result/1', complete=True, input_sha256=input_hash,
                backend='legacy-facemesh/0.10.14', config=asdict(config), model_hashes=models,
                runtime={n: metadata.version(n) for n in ('mediapipe', 'numpy', 'opencv-contrib-python', 'av')},
                timing=dict(mode='decoded-pts/1', audio_stream_start_s=audio_start, audio_decoded_start_s=audio_decoded_start, video_stream_start_s=video_start,
                            anchor_s=anchor, first_video_pts_s=first_pts, last_video_pts_s=last_pts,
                            lip_manual_offset=0., capture_fps=None, encoded_nominal_fps=nominal_rate,
                            decoded_observed_fps=(sequence.index-1)/(last_pts-first_pts) if last_pts>first_pts else None,
                            inference_fps=sequence.index / inference_seconds if inference_seconds else None,
                            processing_fps=sequence.index / elapsed, display_fps=None,
                            dropped_capture_frames=None, decoded_frames=sequence.index),
                coordinates=dict(space='legacy-decoded-pixels', mirrored=False, rotation_applied=any(rotations),
                                 rotation_metadata=rotation,display_rotations=sorted(rotations), resolutions=sorted(dimensions)),
                validity=dict(detected=detected, missing=sequence.index-detected, imputed_measurements=0),
                resource=dict(wall_seconds=elapsed, inference_seconds=inference_seconds, decode_seconds=decode_seconds,
                              output_bytes=sink.written, limits=asdict(limits)),
                smoothing=dict(model='legacy tracking, graph behavior retained', post_filter=config.filter_enabled),
                limitations=['Capture loss cannot be inferred from a file', 'No biological alignment claim',
                             'Decoded PTS differs from legacy frame_index/fps on VFR'])
