"""V2 array orchestration, extracted without changing numeric statements.

Source: speech_synthesis_widget.py; tdklatt: SRC-TDKLATT / REF-KLATT.
UI and path decoding are adapters. See M06-source-map.md for source hashes.
"""
from typing import Optional
from math import gcd
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly
from .klatt_config import *
from .input_parser import VOWEL_FORMANTS, parse_vowel_sequence
from .tdklatt import KlattParam1980, klatt_make
from .spectral_filter import SpectralFilter
from ...acoustic.f0_praat import compute_praat_f0
from ...acoustic.formants_praat import compute_praat_formants
from ...acoustic.energy import compute_energy
from ...acoustic.hnr import compute_hnr
from ...acoustic.shr import compute_shr
from ...acoustic.voicing import compute_voiced_mask
from ...acoustic.jitter_shimmer import compute_jitter_shimmer
from ...acoustic.spectral_slope import compute_spectral_slope
from ...acoustic.spectral_batch import compute_spectral_features_batch

class ParameterCurve:
    def __init__(self, name: str, default_value: float, min_val: float, max_val: float, duration: float):
        self.name = name
        self.default_value = default_value
        self.min_val = min_val
        self.max_val = max_val
        self.points = [(0.0, default_value), (duration, default_value)]
        self.global_override: Optional[float] = None

    def set_points(self, points: list[tuple[float, float]]):
        self.points = sorted(points, key=lambda item: item[0])
        self.global_override = None

    def get_array(self, duration: float, fs: int) -> np.ndarray:
        n_samples = int(round(duration * fs))
        if n_samples <= 0:
            return np.array([], dtype=float)
        if self.global_override is not None:
            return np.ones(n_samples, dtype=float) * float(self.global_override)
        if not self.points:
            return np.ones(n_samples, dtype=float) * self.default_value
        times = [p[0] for p in self.points]
        values = [p[1] for p in self.points]
        if times[0] > 0.0:
            times.insert(0, 0.0)
            values.insert(0, values[0])
        if times[-1] < duration:
            times.append(duration)
            values.append(values[-1])
        t_grid = np.linspace(0.0, duration, n_samples)
        arr = np.interp(t_grid, times, values)
        return np.clip(arr, self.min_val, self.max_val)

class Engine:
    def __init__(self, config):
        self.duration = config['duration']
        self.fs = config['sample_rate']
        self.fade_in = config['fade_in']
        self.fade_out = config['fade_out']
        self.smooth = config['smooth']
        self.sequence = config['sequence']
        self.f0_min_hz, self.f0_max_hz = config['f0_range']
        self.params = {}
        for name, (default, lo, hi, _) in PARAM_DEFAULTS.items():
            if name == 'F0': lo, hi = config['f0_range']
            curve = ParameterCurve(name, default, lo, hi, self.duration)
            curve.points = [tuple(p) for p in config['curves'][name]['points']]
            curve.global_override = config['curves'][name]['override']
            self.params[name] = curve
        self.silence_intervals = [tuple(p) for p in config['silence']]
        self.vowel_boundaries = list(config['boundaries'])

    def _get_neighbor_vowel_formants(self, segments, idx, direction):
        pos = idx + direction
        while 0 <= pos < len(segments):
            seg = segments[pos]
            if seg.type == "vowel":
                return VOWEL_FORMANTS.get(seg.symbol, [500, 1500, 2500])
            pos += direction
        return [500, 1500, 2500]

    def _resize_to(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        if target_len <= 0:
            return np.array([], dtype=float)
        if len(arr) == 0:
            return np.zeros(target_len, dtype=float)
        if len(arr) == target_len:
            return arr.copy()
        x_old = np.linspace(0.0, 1.0, len(arr))
        x_new = np.linspace(0.0, 1.0, target_len)
        return np.interp(x_new, x_old, arr)

    def _slice_array_by_time(self, arr: np.ndarray, start: float, end: float) -> np.ndarray:
        if end <= start or len(arr) == 0:
            return np.array([], dtype=float)
        i0 = int(np.floor(start * self.fs))
        i1 = int(np.ceil(end * self.fs))
        i0 = max(0, min(len(arr), i0))
        i1 = max(0, min(len(arr), i1))
        if i1 <= i0:
            return np.array([], dtype=float)
        return arr[i0:i1].copy()

    def _merge_silence_intervals(self) -> list[tuple[float, float]]:
        if not self.silence_intervals:
            return []
        raw = sorted((max(0.0, float(s)), min(self.duration, float(e))) for s, e in self.silence_intervals if float(e) > float(s))
        if not raw:
            return []
        merged: list[tuple[float, float]] = [raw[0]]
        for s, e in raw[1:]:
            ps, pe = merged[-1]
            if s <= pe + 1e-9:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    def _apply_fade(self, audio: np.ndarray, fade_in_ms: int, fade_out_ms: int) -> np.ndarray:
        if len(audio) == 0:
            return audio
        fade_in_len = int(self.fs * max(0, fade_in_ms) / 1000.0)
        fade_out_len = int(self.fs * max(0, fade_out_ms) / 1000.0)
        if fade_in_len > 0:
            n = min(len(audio), fade_in_len)
            t = np.linspace(0.0, 1.0, n)
            in_env = 0.5 - 0.5 * np.cos(np.pi * t)
            audio[:n] *= in_env
        if fade_out_len > 0:
            n = min(len(audio), fade_out_len)
            t = np.linspace(0.0, 1.0, n)
            out_env = 0.5 + 0.5 * np.cos(np.pi * t)
            audio[-n:] *= out_env
        return audio

    def _compute_rms(self, audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))

    def _match_rms(self, audio: np.ndarray, target_rms: float) -> np.ndarray:
        if len(audio) == 0 or target_rms <= 0.0:
            return audio
        current_rms = self._compute_rms(audio)
        if current_rms <= 1e-10:
            return audio
        gain = float(target_rms / current_rms)
        return audio * gain

    def _synthesize_single_segment(self, arrays: dict[str, np.ndarray], effective_f0: np.ndarray, duration: float) -> np.ndarray:
        if duration <= 0:
            return np.array([], dtype=float)
        klatt_fs = 10000
        kp = KlattParam1980(
            FS=klatt_fs,
            DUR=duration,
            F0=float(effective_f0[0]) if len(effective_f0) else float(self.params["F0"].default_value),
            Jitter=0,
            Shimmer=0,
            SHR=0,
            HNR=None,
            Slope=0,
        )
        n_samp = kp.N_SAMP
        kp.F0 = self._resize_to(effective_f0, n_samp)
        kp.Jitter = self._resize_to(arrays["Jitter"], n_samp)
        kp.Shimmer = np.zeros(n_samp, dtype=float)
        kp.SHR = self._resize_to(arrays["SHR"], n_samp)
        kp.Slope = self._resize_to(arrays["Slope"], n_samp)
        kp.AV = self._resize_to(arrays["AV"], n_samp)
        kp.AVS = np.zeros(n_samp, dtype=float)
        kp.AH = self._resize_to(np.maximum(0.0, 130.0 - arrays["HNR"]), n_samp)
        for idx in range(5):
            f_key = f"F{idx + 1}"
            b_key = f"B{idx + 1}"
            kp.FF[idx] = self._resize_to(arrays[f_key], n_samp)
            kp.BW[idx] = self._resize_to(arrays[b_key], n_samp)
        kp.A1 = self._resize_to(arrays["A1"], n_samp)
        kp.A2 = self._resize_to(arrays["A2"], n_samp)
        kp.A3 = self._resize_to(arrays["A3"], n_samp)
        kp.A4 = self._resize_to(arrays["A4"], n_samp)
        kp.A5 = self._resize_to(arrays["A5"], n_samp)
        engine = klatt_make(kp)
        engine.run()
        output = engine.output
        if output is None or len(output) == 0:
            raise ValueError("Synthesis produced empty output")
        if self.fs != klatt_fs:
            ratio_gcd = gcd(int(self.fs), int(klatt_fs))
            up = int(self.fs) // ratio_gcd
            down = int(klatt_fs) // ratio_gcd
            audio = resample_poly(output, up, down)
        else:
            audio = output
        target_len = int(round(duration * self.fs))
        audio = self._resize_to(audio, target_len)

        def resize_audio(name: str) -> np.ndarray:
            return self._resize_to(arrays[name], target_len)

        spec_filter = SpectralFilter(self.fs)
        audio = spec_filter.process(
            audio,
            self._resize_to(effective_f0, target_len),
            resize_audio("H1H2"),
            resize_audio("Slope"),
            resize_audio("HNR"),
        )
        audio = spec_filter.apply_agc(audio, target_rms=0.1)
        audio = spec_filter.apply_shimmer(audio, resize_audio("Shimmer") * 100.0)
        return spec_filter.normalize(audio)

    def _fit_track_to_len(self, values: Optional[np.ndarray], target_len: int) -> np.ndarray:
        if values is None or target_len <= 0:
            return np.full(max(0, target_len), np.nan, dtype=float)
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return np.full(target_len, np.nan, dtype=float)
        if arr.size == target_len:
            return arr.copy()
        x_new = np.linspace(0.0, 1.0, target_len)
        valid = np.isfinite(arr)
        if np.count_nonzero(valid) == 0:
            return np.full(target_len, np.nan, dtype=float)
        if np.count_nonzero(valid) == 1:
            return np.full(target_len, float(arr[valid][0]), dtype=float)
        x_old = np.linspace(0.0, 1.0, arr.size)[valid]
        y_old = arr[valid]
        return np.interp(x_new, x_old, y_old)

    def _sanitize_track_for_param(self, name: str, values: Optional[np.ndarray], target_len: int) -> np.ndarray:
        arr = self._fit_track_to_len(values, target_len)
        curve = self.params[name]
        if arr.size == 0:
            return np.array([], dtype=float)
        finite = np.isfinite(arr)
        if np.count_nonzero(finite) == 0:
            arr[:] = curve.default_value
        elif np.count_nonzero(finite) < arr.size:
            idx = np.arange(arr.size)
            arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
        return np.clip(arr, curve.min_val, curve.max_val)

    def _apply_track_to_curve(self, name: str, values: Optional[np.ndarray], target_len: int):
        arr = self._sanitize_track_for_param(name, values, target_len)
        if arr.size == 0:
            return
        times = np.linspace(0.0, self.duration, arr.size)
        self.params[name].set_points([(float(t), float(v)) for t, v in zip(times, arr)])

    def generate_vowels(self):
        text = self.sequence.strip()
        if not text:
            return
        segments = parse_vowel_sequence(text)
        if not segments:
            raise ValueError("m06_invalid_sequence")
        weights = [seg.duration_modifier for seg in segments]
        total_weight = sum(weights)
        if total_weight <= 0:
            return
        seg_durs = [self.duration * (w / total_weight) for w in weights]
        boundaries = np.cumsum(seg_durs)[:-1]
        n_grid = max(2, int(round(self.duration * 100)) + 1)
        t_grid = np.linspace(0.0, self.duration, n_grid)
        grids = {
            "F1": np.zeros(n_grid, dtype=float),
            "F2": np.zeros(n_grid, dtype=float),
            "F3": np.zeros(n_grid, dtype=float),
            "AV": np.zeros(n_grid, dtype=float),
        }
        self.silence_intervals = []
        self.vowel_boundaries = [float(x) for x in boundaries]
        av_default = self.params["AV"].default_value
        for idx, seg in enumerate(segments):
            start = 0.0 if idx == 0 else float(boundaries[idx - 1])
            end = self.duration if idx == len(segments) - 1 else float(boundaries[idx])
            mask = (t_grid >= start) & (t_grid <= end)
            if not np.any(mask):
                continue
            if seg.type == "silence":
                self.silence_intervals.append((start, end))
                grids["AV"][mask] = 0.0
                prev_f = self._get_neighbor_vowel_formants(segments, idx, -1)
                next_f = self._get_neighbor_vowel_formants(segments, idx, 1)
                local = np.where(mask)[0]
                alpha = np.linspace(0.0, 1.0, len(local))
                for form_idx, key in enumerate(["F1", "F2", "F3"]):
                    grids[key][mask] = prev_f[form_idx] * (1.0 - alpha) + next_f[form_idx] * alpha
            else:
                formants = VOWEL_FORMANTS.get(seg.symbol, [500, 1500, 2500])
                grids["F1"][mask] = formants[0]
                grids["F2"][mask] = formants[1]
                grids["F3"][mask] = formants[2]
                grids["AV"][mask] = av_default
        smooth_size = self.smooth
        for key in ["F1", "F2", "F3"]:
            grids[key] = uniform_filter1d(grids[key], size=max(1, smooth_size), mode="nearest")
            grids[key] = uniform_filter1d(grids[key], size=max(1, smooth_size), mode="nearest")
        for key in ["F1", "F2", "F3", "AV"]:
            points = [(float(t), float(v)) for t, v in zip(t_grid, grids[key])]
            self.params[key].set_points(points)
        for key in ["F4", "F5"]:
            dv = self.params[key].default_value
            self.params[key].set_points([(0.0, dv), (self.duration, dv)])


    def synthesize(self):
        fade_in_ms = self.fade_in
        fade_out_ms = self.fade_out
        arrays = {name: curve.get_array(self.duration, self.fs) for name, curve in self.params.items()}
        effective_f0 = arrays['F0'].copy()
        mask = arrays['SHR'] >= 0.2
        effective_f0[:len(mask)][mask] *= 2.0
        intervals = self._merge_silence_intervals()
        if intervals:
            pieces: list[np.ndarray] = []
            cursor = 0.0
            reference_rms: Optional[float] = None
            for start, end in intervals:
                if start > cursor:
                    seg_arrays = {name: self._slice_array_by_time(arr, cursor, start) for name, arr in arrays.items()}
                    seg_f0 = self._slice_array_by_time(effective_f0, cursor, start)
                    seg_audio = self._synthesize_single_segment(seg_arrays, seg_f0, start - cursor)
                    seg_audio = self._apply_fade(seg_audio, fade_in_ms, fade_out_ms)
                    seg_rms = self._compute_rms(seg_audio)
                    if reference_rms is None and seg_rms > 1e-10:
                        reference_rms = seg_rms
                    elif reference_rms is not None:
                        seg_audio = self._match_rms(seg_audio, reference_rms)
                    pieces.append(seg_audio)
                silence_len = int(round((end - start) * self.fs))
                if silence_len > 0:
                    pieces.append(np.zeros(silence_len, dtype=float))
                cursor = end
            if cursor < self.duration:
                seg_arrays = {name: self._slice_array_by_time(arr, cursor, self.duration) for name, arr in arrays.items()}
                seg_f0 = self._slice_array_by_time(effective_f0, cursor, self.duration)
                seg_audio = self._synthesize_single_segment(seg_arrays, seg_f0, self.duration - cursor)
                seg_audio = self._apply_fade(seg_audio, fade_in_ms, fade_out_ms)
                if reference_rms is not None:
                    seg_audio = self._match_rms(seg_audio, reference_rms)
                pieces.append(seg_audio)
            if not pieces:
                raise ValueError('Synthesis produced empty output')
            audio = np.concatenate(pieces)
        else:
            audio = self._synthesize_single_segment(arrays, effective_f0, self.duration)
            audio = self._apply_fade(audio, fade_in_ms, fade_out_ms)
        target_total_len = int(round(self.duration * self.fs))
        if len(audio) != target_total_len:
            audio = self._resize_to(audio, target_total_len)
        mx = np.max(np.abs(audio))
        if mx > 1e-08:
            audio = audio / mx * 0.95
        return audio

    def extract(self, audio, mono):
        y = mono
        fs = self.fs
        path = audio
        frameshift_ms = 10.0
        min_f0 = float(self.f0_min_hz)
        max_f0 = float(self.f0_max_hz)
        target_len = max(2, int(round(self.duration * 1000.0 / frameshift_ms)))
        f0 = compute_praat_f0(path, frameshift_ms, min_f0, max_f0, method='cc')
        f0 = self._fit_track_to_len(f0, target_len)
        voiced_mask = compute_voiced_mask(f0)
        formants = compute_praat_formants(path, frameshift_ms, max_formant=6000.0, num_formants=5, pf0=f0)
        f1 = self._fit_track_to_len(formants.get('pF1'), target_len)
        f2 = self._fit_track_to_len(formants.get('pF2'), target_len)
        f3 = self._fit_track_to_len(formants.get('pF3'), target_len)
        f4 = self._fit_track_to_len(formants.get('pF4'), target_len)
        b1 = self._fit_track_to_len(formants.get('pB1'), target_len)
        b2 = self._fit_track_to_len(formants.get('pB2'), target_len)
        b3 = self._fit_track_to_len(formants.get('pB3'), target_len)
        b4 = self._fit_track_to_len(formants.get('pB4'), target_len)
        js = compute_jitter_shimmer(y, fs, frameshift_ms, 160, voiced_mask=voiced_mask, min_f0=min_f0, max_f0=max_f0)
        jitter = self._fit_track_to_len(js.get('Jitter_PPQ5'), target_len)
        shimmer = self._fit_track_to_len(js.get('Shimmer_APQ5'), target_len)
        shr = self._fit_track_to_len(compute_shr(y, fs, frameshift_ms, f0, min_f0, max_f0, voiced_mask=voiced_mask), target_len)
        hnr_res = compute_hnr(y, fs, frameshift_ms, f0, N_periods=5, voiced_mask=voiced_mask)
        hnr_tracks = [self._fit_track_to_len(hnr_res.get(k), target_len) for k in ['HNR05', 'HNR15', 'HNR25', 'HNR35']]
        hnr_stack = np.vstack(hnr_tracks) if hnr_tracks else np.full((1, target_len), np.nan)
        hnr = np.nanmean(hnr_stack, axis=0)
        slope = self._fit_track_to_len(compute_spectral_slope(y, fs, frameshift_ms, f0, min_pitch=min_f0, voiced_mask=voiced_mask), target_len)
        energy = self._fit_track_to_len(compute_energy(y, fs, frameshift_ms, f0, energy_window_ms=20.0), target_len)
        spec = compute_spectral_features_batch(y, fs, frameshift_ms, f0, f1, f2, f3, 5, voiced_mask=voiced_mask)
        a1 = self._fit_track_to_len(spec.get('A1'), target_len)
        a2 = self._fit_track_to_len(spec.get('A2'), target_len)
        a3 = self._fit_track_to_len(spec.get('A3'), target_len)
        h1 = self._fit_track_to_len(spec.get('H1'), target_len)
        h2 = self._fit_track_to_len(spec.get('H2'), target_len)
        h1h2 = h1 - h2
        self._apply_track_to_curve('F0', f0, target_len)
        self._apply_track_to_curve('F1', f1, target_len)
        self._apply_track_to_curve('F2', f2, target_len)
        self._apply_track_to_curve('F3', f3, target_len)
        self._apply_track_to_curve('F4', f4, target_len)
        self._apply_track_to_curve('AV', energy, target_len)
        self._apply_track_to_curve('HNR', hnr, target_len)
        self._apply_track_to_curve('SHR', shr, target_len)
        self._apply_track_to_curve('Jitter', jitter, target_len)
        self._apply_track_to_curve('Shimmer', shimmer, target_len)
        self._apply_track_to_curve('Slope', slope, target_len)
        self._apply_track_to_curve('H1H2', h1h2, target_len)
        self._apply_track_to_curve('A1', a1, target_len)
        self._apply_track_to_curve('A2', a2, target_len)
        self._apply_track_to_curve('A3', a3, target_len)
        self._apply_track_to_curve('B1', b1, target_len)
        self._apply_track_to_curve('B2', b2, target_len)
        self._apply_track_to_curve('B3', b3, target_len)
        self._apply_track_to_curve('B4', b4, target_len)
        for name in ['F5', 'A4', 'A5', 'B5']:
            self.params[name].set_points([(0.0, self.params[name].default_value), (self.duration, self.params[name].default_value)])
