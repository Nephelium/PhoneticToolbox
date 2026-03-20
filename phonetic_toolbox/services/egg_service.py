import numpy as np
from scipy import signal
import warnings
from typing import Optional, Tuple, List
import threading

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False

from phonetic_toolbox.core.egg.analysis import find_gci_goi_peak_min_criterion, calculate_cq_sq
from phonetic_toolbox.core.egg.inverse_filtering import apply_simplified_cp_inverse_filtering
from phonetic_toolbox.core.signals.filters import apply_highpass_filter, apply_lowpass_filter
from phonetic_toolbox.core.acoustic.f0_praat import compute_praat_f0
from phonetic_toolbox.models.egg_models import EGGAnalysisResult
from phonetic_toolbox.models.config import EGGConfig
from phonetic_toolbox.services.io.wav import read_wav
from pathlib import Path
import tempfile
import os
import scipy.io.wavfile as wav

class EGGAnalysisService:
    def __init__(self):
        pass

    def load_file(
        self, 
        filepath: str, 
        config: EGGConfig, 
        flip_channels: bool = False,
        cancel_event: Optional[threading.Event] = None
    ) -> EGGAnalysisResult:
        """
        加载并预处理 EGG 音频文件。
        """
        if cancel_event and cancel_event.is_set(): return None

        # 1. Read WAV
        fs, data = read_wav(filepath)
        
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("WAV 文件必须是立体声（2 声道）。左=EGG，右=音频。")

        # 2. Convert to Float
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            if max_val == 0: max_val = 1
            data = data.astype(np.float32) / max_val
        elif np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        else:
            raise ValueError(f"不支持的数据类型: {data.dtype}")

        if cancel_event and cancel_event.is_set(): return None

        # 3. Normalize Channels Independently (to 0.7)
        # EGG tool logic: normalize each channel to max 0.7
        max_val_0 = float(np.max(np.abs(data[:, 0])))
        if max_val_0 > 0:
            data[:, 0] = (data[:, 0] / max_val_0) * 0.7

        max_val_1 = float(np.max(np.abs(data[:, 1])))
        if max_val_1 > 0:
            data[:, 1] = (data[:, 1] / max_val_1) * 0.7

        # 4. Assign Channels
        egg_channel_idx = 0 if not flip_channels else 1
        audio_channel_idx = 1 if not flip_channels else 0
        egg_signal_raw = data[:, egg_channel_idx]
        audio_signal = data[:, audio_channel_idx]

        if cancel_event and cancel_event.is_set(): return None

        # 5. Preprocessing (Detrend, Filter)
        egg_detrended = signal.detrend(egg_signal_raw)
        egg_highpassed = apply_highpass_filter(egg_detrended, cutoff_freq=config.highpass_cutoff, fs=fs)
        egg_signal_processed = apply_lowpass_filter(egg_highpassed, cutoff_freq=config.lowpass_cutoff, fs=fs)
        
        time_vector = np.arange(len(egg_signal_processed), dtype=float) / float(fs)
        file_duration = float(time_vector[-1]) if len(time_vector) > 0 else 0.0

        # Create Result Object
        result = EGGAnalysisResult(
            time_vector=time_vector,
            egg_signal_raw=egg_signal_raw,
            egg_signal_processed=egg_signal_processed,
            audio_signal=audio_signal,
            fs=fs,
            file_duration=file_duration
        )
        
        return result

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
        use_raw_signal: bool = False
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
             seg_hp = apply_highpass_filter(seg_detrend, cutoff_freq=config.highpass_cutoff, fs=fs)
             seg_lp = apply_lowpass_filter(seg_hp, cutoff_freq=config.lowpass_cutoff, fs=fs)
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
            goi_method=config.goi_method
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
        use_raw_signal: bool = False
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
             seg_hp = apply_highpass_filter(seg_detrend, cutoff_freq=config.highpass_cutoff, fs=fs)
             seg_lp = apply_lowpass_filter(seg_hp, cutoff_freq=config.lowpass_cutoff, fs=fs)
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
            goi_method=config.goi_method
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

    def calculate_praat_f0(self, result: EGGAnalysisResult, cancel_event: Optional[threading.Event] = None):
        if result.audio_signal is None:
            return
            
        if cancel_event and cancel_event.is_set(): return

        try:
            # compute_praat_f0 expects a file path.
            # We must save audio_signal to a temporary file.
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Convert to int16 for wavfile.write if needed, or float32 if supported
            # wavfile.write handles float32 (-1.0 to 1.0)
            wav.write(tmp_path, result.fs, result.audio_signal)
            
            f0_values = compute_praat_f0(
                Path(tmp_path), 
                frameshift_ms=10.0, # Default step
                min_f0=75.0, 
                max_f0=600.0,
                method="ac" # Auto-correlation is standard for Praat To Pitch
            )
            
            # Clean up
            try:
                os.remove(tmp_path)
            except OSError:
                pass
                
            # Create time axis
            # frameshift_ms=10.0 -> time_step=0.01
            time_step = 0.01
            times = np.arange(len(f0_values)) * time_step + (time_step / 2.0) # Center of frame?
            
            result.audio_f0_times = times
            result.audio_f0_values = f0_values
            
        except Exception as e:
            print(f"Praat F0 calculation failed: {e}")

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

    def get_downsampled_data(self, result: EGGAnalysisResult, target_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Helper for timeline plot"""
        if result.egg_signal_processed is None: return None, None
        
        N = len(result.egg_signal_processed)
        if N <= target_points:
            return result.time_vector, result.egg_signal_processed
            
        step = max(1, N // target_points)
        return result.time_vector[::step], result.egg_signal_processed[::step]

    def apply_simplified_cp_inverse_filtering(
        self,
        audio_signal: np.ndarray,
        fs: int,
        gci_times_relative_to_roi_start: np.ndarray,
        lp_order: Optional[int] = None,
        closed_phase_duration_ms: float = 3.0,
        min_segments_for_avg: int = 3,
        tilt_order: int = 1,
        pre_emphasis_alpha: float = 0.97
    ) -> Optional[np.ndarray]:
        """
        Wrapper for core inverse filtering function.
        """
        return apply_simplified_cp_inverse_filtering(
            audio_signal, fs, gci_times_relative_to_roi_start,
            lp_order, closed_phase_duration_ms, min_segments_for_avg,
            tilt_order, pre_emphasis_alpha
        )

    def save_results(self, result: EGGAnalysisResult, filepath: str):
        """
        保存分析结果到文件 (.csv 或 .TextGrid)。
        对于 CSV，保存逐周期的分析结果 (Time, CQ, SQ, F0)。
        """
        if not result or not result.gci_times:
            return

        if filepath.lower().endswith('.csv'):
            try:
                import pandas as pd
                # Ensure we have CQ/SQ
                gci = result.gci_times
                goi = result.goi_times
                peaks = result.peak_times
                
                # Recalculate global CQ/SQ
                times, cq, sq = calculate_cq_sq(gci, goi, peaks)
                
                data = {
                    'Time_s': times,
                    'CQ': cq,
                    'SQ': sq
                }
                
                # Add F0 if available (interpolate to GCI times)
                if result.gci_f0_times is not None and len(result.gci_f0_times) > 0:
                    f0_interp = np.interp(times, result.gci_f0_times, result.gci_f0_values, left=np.nan, right=np.nan)
                    data['F0_Hz'] = f0_interp
                elif result.audio_f0_times is not None and len(result.audio_f0_times) > 0:
                    f0_interp = np.interp(times, result.audio_f0_times, result.audio_f0_values, left=np.nan, right=np.nan)
                    data['F0_Hz'] = f0_interp
                    
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False, float_format='%.6f')
            except ImportError:
                print("Pandas not available for CSV export")
                
        elif filepath.lower().endswith('.textgrid'):
            # Basic TextGrid export using simple text formatting to avoid extra dependencies
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('File type = "ooTextFile"\n')
                    f.write('Object class = "TextGrid"\n\n')
                    f.write('xmin = 0\n')
                    f.write(f'xmax = {result.file_duration}\n')
                    f.write('tiers? <exists>\n')
                    f.write('size = 2\n')
                    f.write('item []:\n')
                    
                    # Tier 1: GCI (PointTier)
                    f.write('    item [1]:\n')
                    f.write('        class = "TextTier"\n')
                    f.write('        name = "GCI"\n')
                    f.write('        xmin = 0\n')
                    f.write(f'        xmax = {result.file_duration}\n')
                    f.write(f'        points: size = {len(result.gci_times)}\n')
                    for i, t in enumerate(result.gci_times):
                        f.write(f'        points [{i+1}]:\n')
                        f.write(f'            number = {t}\n')
                        f.write(f'            mark = "GCI"\n')

                    # Tier 2: GOI (PointTier)
                    f.write('    item [2]:\n')
                    f.write('        class = "TextTier"\n')
                    f.write('        name = "GOI"\n')
                    f.write('        xmin = 0\n')
                    f.write(f'        xmax = {result.file_duration}\n')
                    f.write(f'        points: size = {len(result.goi_times)}\n')
                    for i, t in enumerate(result.goi_times):
                        f.write(f'        points [{i+1}]:\n')
                        f.write(f'            number = {t}\n')
                        f.write(f'            mark = "GOI"\n')
            except Exception as e:
                print(f"TextGrid export failed: {e}")

