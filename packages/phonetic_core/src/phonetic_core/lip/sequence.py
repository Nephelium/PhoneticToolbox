"""M05 versioned per-frame computation. Original formulas/filter stay in their own files."""
from dataclasses import dataclass
import math
import numpy as np
from .metrics import extract_lip_metrics
from .stabilizer import LandmarkStabilizer


@dataclass(frozen=True)
class LipConfig:
    filter_enabled: bool = True
    cutoff_hz: float = 15.0
    method_version: str = 'lip-metrics-v2/1'

    def __post_init__(self):
        if type(self.filter_enabled) is not bool or not math.isfinite(self.cutoff_hz) or not 1 <= self.cutoff_hz <= 240:
            raise ValueError('invalid_lip_filter')
        if self.method_version != 'lip-metrics-v2/1':
            raise ValueError('unsupported_lip_method')


def mesh_neighbors(connections, count=478):
    neighbors = [set() for _ in range(count)]
    for a, b in connections:
        if 0 <= a < count and 0 <= b < count:
            neighbors[a].add(b)
            neighbors[b].add(a)
    return [np.array(sorted(n), dtype=np.int64) for n in neighbors]


def json_numbers(values):
    if isinstance(values, np.ndarray): return json_numbers(values.tolist())
    if isinstance(values, dict): return {key: json_numbers(value) for key, value in values.items()}
    if isinstance(values, (list, tuple)): return [json_numbers(value) for value in values]
    if isinstance(values, (float, np.floating)):
        return float(values) if math.isfinite(values) else None
    return values


class LipSequence:
    def __init__(self, config=LipConfig(), neighbors=None):
        self.config = config
        self.filter = LandmarkStabilizer(min_cutoff_hz=config.cutoff_hz, neighbor_indices=neighbors)
        self.previous_time = None
        self.index = 0

    def process(self, points, time_s):
        if not math.isfinite(time_s) or (self.previous_time is not None and time_s <= self.previous_time):
            raise ValueError('nonmonotonic_frame_time')
        if points is not None:
            points = np.asarray(points, dtype=np.float32)
            if points.shape != (478, 2) or not np.isfinite(points).all():
                raise ValueError('invalid_landmarks')
            processed = self.filter.filter(points, time_s) if self.config.filter_enabled else points.copy()
            raw_metrics = extract_lip_metrics(points)
            metrics = extract_lip_metrics(processed)
        else:
            processed = raw_metrics = metrics = None
        row = json_numbers(dict(index=self.index, time_s=float(time_s), detected=points is not None,
                                reason=None if points is not None else 'face_not_detected',
                                raw_points=points, points=processed, raw_metrics=raw_metrics, metrics=metrics))
        self.previous_time = time_s
        self.index += 1
        return row


def offset_time(time_s, offset_s, action='apply'):
    if action == 'cancel': return None
    if action not in ('apply', 'save_without_offset') or not math.isfinite(offset_s) or not -2 <= offset_s <= 2:
        raise ValueError('invalid_offset')
    return time_s + (offset_s if action == 'apply' else 0.)
