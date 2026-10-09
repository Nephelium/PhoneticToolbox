"""M03 local migration of v2 EGG behavior. Source: PENDING-EGG; provenance unresolved.
See NOTICE.txt. Pure numerical compatibility layer, no file/device/GUI access.
"""
import threading
import numpy as np
from scipy import signal
from typing import Optional, Tuple, List
from .config import EGGConfig
from .model import EGGAnalysisResult
from .events import find_gci_goi_peak_min_criterion
from .metrics import calculate_cq_sq
from .filters import apply_highpass_filter, apply_lowpass_filter

class LegacyCalculations:
    def analyze_events(
        self,
        result: EGGAnalysisResult,
        config: EGGConfig,
        cancel_event: Optional[threading.Event] = None
    ) -> EGGAnalysisResult:
        """
        计算 GCI, GOI, Peak, CQ, SQ。
        """
        if result is None or result.egg_signal_processed is None:
            return result

        if cancel_event and cancel_event.is_set(): return result

        # Auto Prominence Logic
        peak_prom = config.peak_prominence
        if config.auto_prominence:
            # Simple global auto prominence heuristic if needed,
            # but find_gci_goi... handles local prominence if use_local_prominence=True
            pass # logic inside core function

        gci, goi, peaks = find_gci_goi_peak_min_criterion(
            result.egg_signal_processed,
            result.fs,
            min_f0=50, max_f0=500, # Could be config
            criterion_level=config.criterion_level,
            peak_prominence=peak_prom,
            valley_prominence=config.valley_prominence,
            use_local_prominence=config.auto_prominence,
            local_window_s=0.2, local_hop_s=0.1, min_auto_prom=config.min_auto_prominence,
            gci_method=config.gci_method,
            goi_method=config.goi_method,
            cancel_event=cancel_event
        )

        if cancel_event and cancel_event.is_set(): return result

        result.gci_times = gci
        result.goi_times = goi
        result.peak_times = peaks

        # Calculate GCI-F0
        self._calculate_gci_f0(result)

        # Calculate CQ/SQ (Global) - Though usually done per ROI in GUI for speed
        # But if we want global stats, we can do it here.
        # Note: main_app.py does global GCI/GOI but local CQ/SQ for plots.
        # We can calculate global CQ/SQ here if it's fast enough.
        # cq_t, cq_v, sq_v = calculate_cq_sq(gci, goi, peaks)
        # result.cq_times = cq_t
        # result.cq_values = cq_v
        # result.sq_values = sq_v

        return result

    def calculate_cq_sq_segment(
        self,
        result: EGGAnalysisResult,
        start_s: float,
        end_s: float,
        config: EGGConfig,
        use_raw_signal: bool = False,
        cancel_event = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate CQ/SQ for a specific segment (ROI).
        """
        fs = result.fs
        start_idx = max(0, int((start_s - 0.1) * fs)) # 100ms padding

        # Determine signal source
        full_signal = result.egg_signal_raw if use_raw_signal else result.egg_signal_processed

        end_idx = min(len(full_signal), int((end_s + 0.1) * fs))

        if start_idx >= end_idx:
            return None, None, None

        segment = full_signal[start_idx:end_idx]
        offset_s = start_idx / fs

        # If using raw signal, we might need to apply highpass filter locally to get meaningful peaks?
        # But user asked to use "displayed waveform".
        # Raw waveform is usually DC-offset and drifty.
        # If user switches to "Raw", they see raw.
        # If we compute metrics on raw, they might be bad if not filtered.
        # However, user explicit request: "CQ、SQ、GCI、GOI根据我实时波形来计算"
        # So if display is Raw, we use Raw.

        # Note: Raw signal usually needs at least detrending/highpass for GCI/GOI to work well.
        # But we follow user instruction strictly.

        # BUT, if we use Raw, we should probably at least detrend locally if it's very drifty,
        # otherwise threshold-based methods fail.
        # The EGGService.load_file does: detrend -> highpass -> lowpass for 'processed'.
        # 'raw' is just normalized 0-1.

        # If we want to support slider changes (highpass), we need to re-filter the raw signal
        # using the NEW cutoff from config.
        # The 'result.egg_signal_processed' was computed with OLD config at load time.
        # So we MUST re-compute 'processed' from 'raw' if we want slider to affect it.

        # Let's change the strategy:
        # Instead of just picking raw vs processed, we should:
        # 1. Always take 'egg_signal_raw'.
        # 2. If use_raw_signal is True (Display: Raw), use it as is.
        # 3. If use_raw_signal is False (Display: Filtered), apply filters with CURRENT config.

        if not use_raw_signal:
             # Apply filters on the segment? Or on the whole file?
             # Applying on segment might have edge artifacts.
             # Applying on whole file is slow for slider drag.
             # For a "slider drag" experience, maybe segment is enough if we have padding.
             # We already added 100ms padding.

             # Re-process segment
             seg_detrend = signal.detrend(segment)
             stable = result.method_version == 'egg-bounded/2'
             seg_hp = apply_highpass_filter(seg_detrend, cutoff_freq=config.highpass_cutoff, fs=fs, stable=stable)
             seg_lp = apply_lowpass_filter(seg_hp, cutoff_freq=config.lowpass_cutoff, fs=fs, stable=stable)
             segment_to_analyze = seg_lp
        else:
             segment_to_analyze = segment

        gci, goi, peaks = find_gci_goi_peak_min_criterion(
            segment_to_analyze, fs,
            min_f0=50, max_f0=500,
            criterion_level=config.criterion_level,
            peak_prominence=config.peak_prominence,
            valley_prominence=config.valley_prominence,
            use_local_prominence=config.auto_prominence,
            local_window_s=0.2, local_hop_s=0.1, min_auto_prom=config.min_auto_prominence,
            gci_method=config.gci_method,
            goi_method=config.goi_method,
            cancel_event=cancel_event
        )

        # Adjust times
        if gci: gci = [t + offset_s for t in gci]
        if goi: goi = [t + offset_s for t in goi]
        if peaks: peaks = [t + offset_s for t in peaks]

        return calculate_cq_sq(gci, goi, peaks)

    def get_events_segment(
        self,
        result: EGGAnalysisResult,
        start_s: float,
        end_s: float,
        config: EGGConfig,
        use_raw_signal: bool = False,
        cancel_event = None
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Get GCI, GOI, and Peaks for a specific segment using current config.
        """
        fs = result.fs
        start_idx = max(0, int((start_s - 0.05) * fs)) # 50ms padding

        full_signal = result.egg_signal_raw # Always start from raw

        end_idx = min(len(full_signal), int((end_s + 0.05) * fs))

        if start_idx >= end_idx:
            return [], [], []

        segment = full_signal[start_idx:end_idx]
        offset_s = start_idx / fs

        # Apply filters if needed (Display: Filtered)
        if not use_raw_signal:
             seg_detrend = signal.detrend(segment)
             stable = result.method_version == 'egg-bounded/2'
             seg_hp = apply_highpass_filter(seg_detrend, cutoff_freq=config.highpass_cutoff, fs=fs, stable=stable)
             seg_lp = apply_lowpass_filter(seg_hp, cutoff_freq=config.lowpass_cutoff, fs=fs, stable=stable)
             segment_to_analyze = seg_lp
        else:
             segment_to_analyze = segment

        peak_prom = config.peak_prominence

        gci, goi, peaks = find_gci_goi_peak_min_criterion(
            segment_to_analyze, fs,
            min_f0=50, max_f0=500,
            criterion_level=config.criterion_level,
            peak_prominence=peak_prom,
            valley_prominence=config.valley_prominence,
            use_local_prominence=config.auto_prominence,
            local_window_s=0.2, local_hop_s=0.1, min_auto_prom=config.min_auto_prominence,
            gci_method=config.gci_method,
            goi_method=config.goi_method,
            cancel_event=cancel_event
        )

        # Adjust times
        if gci: gci = [t + offset_s for t in gci]
        if goi: goi = [t + offset_s for t in goi]
        if peaks: peaks = [t + offset_s for t in peaks]

        return gci, goi, peaks

    def _calculate_gci_f0(self, result: EGGAnalysisResult):
        if not result.gci_times or len(result.gci_times) < 2:
            result.gci_f0_times = None
            result.gci_f0_values = None
            return

        try:
            gci_np = np.array(result.gci_times)
            periods = np.diff(gci_np)
            f0_values = 1.0 / periods
            f0_times = gci_np[:-1] + periods / 2.0

            vals = np.array(f0_values, dtype=float)
            times = np.array(f0_times, dtype=float)

            if len(vals) > 2:
                med = np.median(vals)
                mad = np.median(np.abs(vals - med))
                thr = 3.0 * mad if mad > 0 else max(1e-12, 3.0 * np.std(vals))
                base = vals.copy()
                base[np.abs(base - med) >= thr] = np.nan

                force_keep = vals < 100.0
                base[force_keep] = vals[force_keep]

                nan_arr = np.isnan(base)
                left_nan = np.concatenate(([False], nan_arr[:-1]))
                right_nan = np.concatenate((nan_arr[1:], [False]))
                keep = (force_keep) | ((~nan_arr) & (~(left_nan & right_nan)))

                result.gci_f0_times = times[keep]
                result.gci_f0_values = base[keep]
            else:
                result.gci_f0_times = times
                result.gci_f0_values = vals
        except Exception:
            result.gci_f0_times = None
            result.gci_f0_values = None

    def detect_glottal_movement(self, result: EGGAnalysisResult):
        if result.audio_f0_values is None or len(result.audio_f0_values) < 2:
            result.glottal_movement_events = []
            return

        f0_vals = result.audio_f0_values
        f0_times = result.audio_f0_times

        candidates = []
        SLOPE_THRESHOLD = 1000.0

        dt = np.diff(f0_times)
        df0 = np.diff(f0_vals)

        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = df0 / dt

        valid_indices = np.where(~np.isnan(slopes))[0]

        for i in valid_indices:
            slope = slopes[i]
            t_start = f0_times[i]
            if slope > SLOPE_THRESHOLD:
                candidates.append((t_start, "Rise"))
            elif slope < -SLOPE_THRESHOLD:
                candidates.append((t_start, "Fall"))

        # Filter
        final_events = []

        # Rise
        rise_events = sorted([x for x in candidates if x[1] == "Rise"], key=lambda x: x[0])
        last_rise = -1.0
        for t, m in rise_events:
            if last_rise < 0 or (t - last_rise >= 0.1):
                final_events.append((t, m))
                last_rise = t

        # Fall
        fall_events = sorted([x for x in candidates if x[1] == "Fall"], key=lambda x: x[0])
        last_fall = -1.0
        for t, m in fall_events:
            if last_fall < 0 or (t - last_fall >= 0.1):
                final_events.append((t, m))
                last_fall = t

        final_events.sort(key=lambda x: x[0])
        result.glottal_movement_events = final_events
