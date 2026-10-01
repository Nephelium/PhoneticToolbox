# M01-B source_ids: SRC-PRAAT, SRC-REAPER, SRC-IRAPT, SRC-WMPC, SRC-VOICESAUCE, SRC-OPENSAUCE; migration evidence: docs/modules/evidence/M01-core-migration.json
from dataclasses import asdict
from typing import Optional, List
import numpy as np
import pandas as pd
from ..models.acoustic import AcousticConfig, AnalysisResult
from ..models.audio import AudioInput
from ..models.associations import AcousticAssociations
from ..ports.acoustic import AcousticBackends
from ..ports.errors import raise_stage_failure
from ..acoustic.catalog import CORE_RESULT_FIELD_MAP
from ..acoustic.alignment import align_track_to_grid, smooth_preserving_gaps
from ..acoustic.lip import interpolate_lip
from ..acoustic.annotations import align_annotations
from ..acoustic import (
    compute_praat_f0_track,
    compute_praat_formants,
    compute_spectral_features_batch,
    compute_cpp,
    compute_hnr,
    compute_shr,
    compute_jitter_shimmer,
    compute_spectral_slope,
    compute_soe,
    correct_formants,
    compute_H1A1A2A3_corrected,
    compute_H1H2_H2H4_corrected,
    compute_corrections_H2KH5K,
    compute_energy,
    compute_silence_mask,
    compute_voiced_mask
)

def analyze_audio(audio: AudioInput, config: AcousticConfig, associations=None, backends=None, cancellation=None) -> AnalysisResult:
    """Array-only M01 computation. Native execution and persistence belong to adapters."""
    if not isinstance(audio,AudioInput): raise TypeError('audio must be AudioInput')
    backends=backends or AcousticBackends()
    associations=associations or AcousticAssociations()
    backend_events=[]
    def checkpoint():
        if cancellation is not None: cancellation()
    checkpoint()
    y=audio.scientific_mono()
    fs=audio.sample_rate_hz
    if not len(y):
        # The array API keeps an empty result for empty input. Task publication
        # rejects empty audio before calling this API; no algorithm is invoked.
        return AnalysisResult(time_axis=np.array([],dtype=float),f0_praat=np.array([],dtype=float),
            sampling_rate=fs,config_snapshot=asdict(config))
    wav_path='<decoded audio>'
    # Use config object properties
    frameshift_ms = config.frameshift_ms
    min_f0 = config.min_f0
    max_f0 = config.max_f0
    n_periods = config.n_periods

    # --- 0. Calculate Intensity & Silence Mask ---
    # User requested: "silence threshold" default 0.03.
    # Energy lower than this threshold -> all params set to NaN.
    # "silence_threshold" (0.03) usually means relative to maximum intensity (amplitude ratio).
    # We calculate Intensity (dB) first.
    try:
        # We use pF0 for framing alignment if possible, but we don't have it yet.
        # So we pass None for F0, compute_energy will generate frames based on length.
        # Or better, we generate dummy F0 or just rely on frameshift.
        # compute_energy expects F0 to determine length.
        # Let's estimate length.
        est_len = int(len(y) / (fs * frameshift_ms / 1000.0))
        dummy_f0 = np.zeros(est_len)

        intensity = compute_energy(y, fs, frameshift_ms, dummy_f0, energy_window_ms=config.energy_window_ms)

        # Determine silence threshold in dB using new core function
        silence_mask = compute_silence_mask(intensity, config.silence_threshold)

    except Exception as e:
        raise_stage_failure('energy', e)

    checkpoint()
    # --- 1. 计算 Formants (Praat Burg) ---
    try:
        formant_res = compute_praat_formants(
            audio,
            frameshift_ms,
            max_formant=config.max_formant,
            num_formants=config.num_formants
        )
    except Exception as e:
        raise_stage_failure('formants', e)

    # 获取共振峰数据供后续使用
    pF1 = formant_res.get("pF1")
    pF2 = formant_res.get("pF2")
    pF3 = formant_res.get("pF3")
    pF4 = formant_res.get("pF4")
    pB1 = formant_res.get("pB1")
    pB2 = formant_res.get("pB2")
    pB3 = formant_res.get("pB3")
    pB4 = formant_res.get("pB4")

    target_len = len(pF1)
    target_times = np.arange(target_len, dtype=float) * frameshift_ms / 1000.0

    checkpoint()
    # --- 3. 计算 F0 (Praat & REAPER) ---
    f0_data = {}

    # 3.1 Praat F0
    try:
        pf0_track = compute_praat_f0_track(
            audio,
            frameshift_ms,
            min_f0,
            max_f0,
            "cc",
        )
        f0_data["pF0"] = align_track_to_grid(
            pf0_track.times,
            pf0_track.values,
            target_times,
        )
    except Exception as e:
        raise_stage_failure('praat_pitch', e)

    # 3.2 REAPER F0 (Optional)
    try:
        if config.use_reaper:
            # REAPER expects seconds, config has ms
            rf0_res = backends.reaper(
                audio,
                frameshift_ms / 1000.0,
                min_f0,
                max_f0,
                hilbert=config.reaper_hilbert,
                no_highpass=config.reaper_no_highpass,
            )
            backend_events.append({"stage":"reaper","actual":rf0_res.actual_backend,"reason":rf0_res.reason})
            f0_data["rF0"] = align_track_to_grid(
                rf0_res.times,
                rf0_res.values,
                target_times,
            )
        else:
            f0_data["rF0"] = np.full(target_len, np.nan)
    except Exception as e:
        raise_stage_failure('reaper', e)

    checkpoint()
    # --- 4. 基于 F0 计算衍生参数 (Harmonics, Amplitudes, Corrections, etc.) ---
    # 我们需要为 pF0 和 rF0 分别计算一套参数

    derived_data = {}

    for f0_type in ["pF0", "rF0"]:
        checkpoint()
        f0 = f0_data[f0_type]
        suffix = f"_{f0_type}" # e.g. _pF0

        # Voiced Mask (using new core function)
        # Combine F0 detection with Silence detection
        # Note: We need to ensure silence_mask matches length of f0
        # f0 length is target_len

        # Pad or truncate silence_mask to match target_len
        curr_silence_mask = None
        if silence_mask is not None:
            if len(silence_mask) >= target_len:
                curr_silence_mask = silence_mask[:target_len]
            else:
                curr_silence_mask = np.pad(silence_mask, (0, target_len - len(silence_mask)), constant_values=True)

        voiced_mask = compute_voiced_mask(f0, curr_silence_mask)

        # 4.1 Harmonics (H1, H2, H4) and Amplitudes (A1, A2, A3) and H2K, H5K
        # Use new Batch Computation
        try:
            spec_res = compute_spectral_features_batch(
                y, fs, frameshift_ms, f0, pF1, pF2, pF3, n_periods, voiced_mask
            )
            # Unpack
            h1 = spec_res["H1"]
            h2 = spec_res["H2"]
            h4 = spec_res["H4"]
            a1 = spec_res["A1"]
            a2 = spec_res["A2"]
            a3 = spec_res["A3"]
            h2k = spec_res["H2K"]
            h5k = spec_res["H5K"]

            # Store raw
            derived_data[f"H1{suffix}"] = h1
            derived_data[f"H2{suffix}"] = h2
            derived_data[f"H4{suffix}"] = h4
            derived_data[f"A1{suffix}"] = a1
            derived_data[f"A2{suffix}"] = a2
            derived_data[f"A3{suffix}"] = a3
            derived_data[f"H2K{suffix}"] = h2k
            derived_data[f"H5K{suffix}"] = h5k

        except Exception as e:
            raise_stage_failure('spectrum', e)

        # 4.3 Uncorrected Tilts (H1-H2, H1-A1, etc.)
        # "H1H2u" means uncorrected
        try:
            derived_data[f"H1H2u{suffix}"] = h1 - h2
            derived_data[f"H2H4u{suffix}"] = h2 - h4
            derived_data[f"H1A1u{suffix}"] = h1 - a1
            derived_data[f"H1A2u{suffix}"] = h1 - a2
            derived_data[f"H1A3u{suffix}"] = h1 - a3
            derived_data[f"H42Ku{suffix}"] = h4 - h2k
            derived_data[f"H2KH5Ku{suffix}"] = h2k - h5k
        except Exception as e:
            raise_stage_failure('tilt', e)

        # 4.4 Corrected Tilts
        try:
            h_corr = compute_H1H2_H2H4_corrected(h1, h2, h4, fs, f0, pF1, pF2, pB1, pB2)
            derived_data[f"H1H2c{suffix}"] = h_corr["H1H2c"]
            derived_data[f"H2H4c{suffix}"] = h_corr["H2H4c"]

            a_corr = compute_H1A1A2A3_corrected(h1, a1, a2, a3, fs, f0, pF1, pF2, pF3, pB1, pB2, pB3)
            derived_data[f"H1A1c{suffix}"] = a_corr["H1A1c"]
            derived_data[f"H1A2c{suffix}"] = a_corr["H1A2c"]
            derived_data[f"H1A3c{suffix}"] = a_corr["H1A3c"]

            # Corrected 2K, 5K using Iseli correction
            # Ensure F4/B4 are arrays (might be None if num_formants < 4)
            f4_arr = pF4 if pF4 is not None else np.full(target_len, np.nan)
            b4_arr = pB4 if pB4 is not None else np.full(target_len, np.nan)

            corr_res = compute_corrections_H2KH5K(
                h4, h2k, h5k, int(fs), f0,
                pF1, pF2, pF3, f4_arr,
                pB1, pB2, pB3, b4_arr
            )

            derived_data[f"H42Kc{suffix}"] = corr_res["H42Kc"]
            derived_data[f"H2KH5Kc{suffix}"] = corr_res["H2KH5Kc"]
        except Exception as e:
            raise_stage_failure('correction', e)

        # 4.6 CPP
        try:
            cpp_val = compute_cpp(y, fs, frameshift_ms, f0, n_periods, voiced_mask)
            derived_data[f"CPP{suffix}"] = cpp_val
        except Exception as e:
            raise_stage_failure('cpp', e)

        # 4.7 HNR
        try:
            hnr_res = compute_hnr(y, fs, frameshift_ms, f0, n_periods, voiced_mask=voiced_mask)
            # output_text.py expects HNR05_pF0, HNR15_pF0...
            for hk, hv in hnr_res.items():
                # hk is like HNR05, HNR15
                derived_data[f"{hk}{suffix}"] = hv[:target_len]
        except Exception as e:
            raise_stage_failure('hnr', e)

        # 4.8 SHR
        try:
            shr_val = compute_shr(y, fs, frameshift_ms, f0, min_f0, max_f0, voiced_mask=voiced_mask)
            derived_data[f"SHR{suffix}"] = shr_val[:target_len]
        except Exception as e:
            raise_stage_failure('shr', e)

        # 4.9 Spectral Slope
        try:
            slope = compute_spectral_slope(y, fs, frameshift_ms, f0, min_pitch=min_f0, voiced_mask=voiced_mask)
            derived_data[f"SpectralSlope{suffix}"] = slope[:target_len]
        except Exception as e:
            raise_stage_failure('slope', e)

        # 4.10 SOE (Strength of Excitation)
        try:
            # Use compute_soe which uses ZFF
            # compute_soe returns (soe_array, epoch_indices)
            soe_val, _ = compute_soe(y, fs, frameshift_ms, f0, target_len)
            derived_data[f"SOE{suffix}"] = soe_val
        except Exception as e:
            raise_stage_failure('soe', e)

    checkpoint()
    # --- 5. Global Parameters (Intensity, Jitter, Shimmer) ---
    # Intensity
    # Already calculated at step 0
    derived_data["Intensity"] = intensity[:target_len]

    # Jitter/Shimmer (Using Praat pF0 logic usually)
    try:
        # compute_jitter_shimmer uses Praat internally
        # Use larger window to ensure enough pulses for APQ11 (needs 11 periods)
        # For min_f0=75Hz, period is ~13.3ms. 11 periods ~ 147ms.
        # We use max(160, config.windowsize_ms) to be safe for low pitch.
        js_win = max(160, config.windowsize_ms)
        js_res = compute_jitter_shimmer(y, fs, frameshift_ms, js_win, voiced_mask=(f0_data["pF0"] > 0), min_f0=min_f0, max_f0=max_f0, f0_provider=backends.wm_f0, backend_events=backend_events)
        for k, v in js_res.items():
            derived_data[k] = v[:target_len]
    except Exception as e:
        raise_stage_failure('jitter_shimmer', e)

    # CPP (Generic) - usually copy CPP_pF0
    if "CPP_pF0" in derived_data:
        pass
        # derived_data["CPP"] = derived_data["CPP_pF0"] # Removed as per user request (duplicate)

    checkpoint()
    # Already parsed associations; no filesystem or pickle operations.
    lip_data = interpolate_lip(associations.lip, target_times, companion_start=associations.lip_companion_start) if associations.lip is not None else {}
    textgrid_data = align_annotations(associations.tiers, target_len, frameshift_ms)

    # --- 8. Construct AnalysisResult ---
    parameters = {}
    parameters.update(derived_data)

    # Helper to ensure length matches target_len
    def fix_len(arr, fill_val=np.nan):
        if arr is None: return None
        if len(arr) == target_len: return arr
        if len(arr) > target_len: return arr[:target_len]
        # Pad
        if np.issubdtype(arr.dtype, np.number):
            return np.pad(arr, (0, target_len - len(arr)), constant_values=fill_val)
        else:
            # For object/string arrays
            new_arr = np.full(target_len, fill_val, dtype=arr.dtype)
            new_arr[:len(arr)] = arr
            return new_arr

    # Apply fix_len to all data sources
    f0_praat = fix_len(f0_data.get("pF0"))
    f0_reaper = fix_len(f0_data.get("rF0"))

    f1 = fix_len(formant_res.get("pF1"))
    f2 = fix_len(formant_res.get("pF2"))
    f3 = fix_len(formant_res.get("pF3"))
    f4 = fix_len(formant_res.get("pF4"))

    b1 = fix_len(formant_res.get("pB1"))
    b2 = fix_len(formant_res.get("pB2"))
    b3 = fix_len(formant_res.get("pB3"))
    b4 = fix_len(formant_res.get("pB4"))

    intensity_fixed = fix_len(intensity[:target_len])

    for k in parameters:
        parameters[k] = fix_len(parameters[k])

    for k in lip_data:
        lip_data[k] = fix_len(lip_data[k])

    for k in textgrid_data:
        textgrid_data[k] = fix_len(textgrid_data[k], fill_val="")

    result = AnalysisResult(
        time_axis=target_times,
        sampling_rate=fs,  # M01-D01: describe the input, not the REAPER intermediate.

        f0_praat=f0_praat,
        f0_reaper=f0_reaper,

        f1=f1, f2=f2, f3=f3, f4=f4,
        b1=b1, b2=b2, b3=b3, b4=b4,

        intensity=intensity_fixed,

        parameters=parameters,
        lip_data=lip_data,
        textgrid_data=textgrid_data
    )

    # --- 8.5 Apply Silence/Voicing Mask ---
    final_mask = None

    if config.only_voiced:
        praat_voiced = np.zeros(target_len, dtype=bool)
        reaper_voiced = np.zeros(target_len, dtype=bool)
        if f0_praat is not None:
            praat_voiced = np.isfinite(f0_praat) & (f0_praat > 0)
        if f0_reaper is not None:
            reaper_voiced = np.isfinite(f0_reaper) & (f0_reaper > 0)
        voiced_any = praat_voiced | reaper_voiced
        final_mask = ~voiced_any

    else:
        if len(silence_mask) >= target_len:
            final_mask = silence_mask[:target_len]
        else:
            final_mask = np.pad(silence_mask, (0, target_len - len(silence_mask)), constant_values=True)

    # Apply mask
    if final_mask is not None:
        # Apply masking to AnalysisResult fields
        if result.f0_praat is not None: result.f0_praat[final_mask] = np.nan
        if result.f0_reaper is not None: result.f0_reaper[final_mask] = np.nan

        for attr in ['f1', 'f2', 'f3', 'f4', 'b1', 'b2', 'b3', 'b4']:
            val = getattr(result, attr)
            if val is not None:
                val[final_mask] = np.nan

        for k, v in result.parameters.items():
            if np.issubdtype(v.dtype, np.number):
                 v[final_mask] = np.nan

    # --- 9. Smoothing (Optional) ---
    if config.smooth_win_size > 1:
        win_size = config.smooth_win_size

        if result.f0_praat is not None:
            result.f0_praat = smooth_preserving_gaps(result.f0_praat, win_size)
        if result.f0_reaper is not None:
            result.f0_reaper = smooth_preserving_gaps(result.f0_reaper, win_size)

        for attr in ['f1', 'f2', 'f3', 'f4', 'b1', 'b2', 'b3', 'b4']:
            val = getattr(result, attr)
            if val is not None:
                setattr(result, attr, smooth_preserving_gaps(val, win_size))

        for k, v in result.parameters.items():
            if np.issubdtype(v.dtype, np.number):
                 result.parameters[k] = smooth_preserving_gaps(v, win_size)

        # Smoothing must never restore frames rejected by the active mask.
        if final_mask is not None:
            if result.f0_praat is not None:
                result.f0_praat[final_mask] = np.nan
            if result.f0_reaper is not None:
                result.f0_reaper[final_mask] = np.nan
            for attr in ['f1', 'f2', 'f3', 'f4', 'b1', 'b2', 'b3', 'b4']:
                val = getattr(result, attr)
                if val is not None:
                    val[final_mask] = np.nan
            for value in result.parameters.values():
                if np.issubdtype(value.dtype, np.number):
                    value[final_mask] = np.nan

    # --- 10. Lip Smoothing (Optional, separate) ---
    if config.lip_smooth_win_size > 1 and result.lip_data:
        lip_win = config.lip_smooth_win_size
        def smooth_lip(arr):
            if arr is None or len(arr) == 0:
                return arr
            return pd.Series(arr).rolling(window=lip_win, center=True, min_periods=1).mean().values
        for k, v in result.lip_data.items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                result.lip_data[k] = smooth_lip(v)

    checkpoint()
    result.backend_events=backend_events
    result.config_snapshot=asdict(config)
    return _apply_selected_parameters(result, config.selected_parameter_keys)

def _apply_selected_parameters(result: AnalysisResult, selected_keys: Optional[List[str]]) -> AnalysisResult:
    if not selected_keys:
        return result
    selected = set(selected_keys)
    if "Energy" in selected and "Intensity" not in selected:
        selected.add("Intensity")
    for key, attr in CORE_RESULT_FIELD_MAP.items():
        if key not in selected and hasattr(result, attr):
            setattr(result, attr, None)
    result.parameters = {k: v for k, v in result.parameters.items() if k in selected}
    result.lip_data = {k: v for k, v in result.lip_data.items() if k in selected}
    return result
