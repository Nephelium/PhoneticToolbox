import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
import traceback
import logging

from phonetic_toolbox.models.config import AcousticConfig, AnalysisResult
from phonetic_toolbox.core.acoustic import (
    compute_praat_f0,
    compute_reaper_f0,
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
from phonetic_toolbox.services.io.lip import read_lip_data
from phonetic_toolbox.services.io.excel import save_excel, save_fast_parameter_db
from phonetic_toolbox.services.io.textgrid import parse_textgrid
import scipy.io.wavfile as wavfile
import warnings

PARAMETER_MAPPING = {
    "pF0": "F0 - Praat",
    "rF0": "F0 - REAPER",
    "pF1": "F1 - Praat",
    "pF2": "F2 - Praat",
    "pF3": "F3 - Praat",
    "pF4": "F4 - Praat",
    "pB1": "B1 - Praat",
    "pB2": "B2 - Praat",
    "pB3": "B3 - Praat",
    "pB4": "B4 - Praat",
    "H1_pF0": "H1 (pF0)",
    "H1_rF0": "H1 (rF0)",
    "H2_pF0": "H2 (pF0)",
    "H2_rF0": "H2 (rF0)",
    "H4_pF0": "H4 (pF0)",
    "H4_rF0": "H4 (rF0)",
    "A1_pF0": "A1 (pF0)",
    "A1_rF0": "A1 (rF0)",
    "A2_pF0": "A2 (pF0)",
    "A2_rF0": "A2 (rF0)",
    "A3_pF0": "A3 (pF0)",
    "A3_rF0": "A3 (rF0)",
    "H1H2u_pF0": "H1-H2 (pF0)",
    "H1H2u_rF0": "H1-H2 (rF0)",
    "H2H4u_pF0": "H2-H4 (pF0)",
    "H2H4u_rF0": "H2-H4 (rF0)",
    "H1A1u_pF0": "H1-A1 (pF0)",
    "H1A1u_rF0": "H1-A1 (rF0)",
    "H1A2u_pF0": "H1-A2 (pF0)",
    "H1A2u_rF0": "H1-A2 (rF0)",
    "H1A3u_pF0": "H1-A3 (pF0)",
    "H1A3u_rF0": "H1-A3 (rF0)",
    "H1A1c_pF0": "H1*-A1* (pF0)",
    "H1A1c_rF0": "H1*-A1* (rF0)",
    "H1A2c_pF0": "H1*-A2* (pF0)",
    "H1A2c_rF0": "H1*-A2* (rF0)",
    "H1A3c_pF0": "H1*-A3* (pF0)",
    "H1A3c_rF0": "H1*-A3* (rF0)",
    "H1H2c_pF0": "H1*-H2* (pF0)",
    "H1H2c_rF0": "H1*-H2* (rF0)",
    "H2H4c_pF0": "H2*-H4* (pF0)",
    "H2H4c_rF0": "H2*-H4* (rF0)",
    "H2K_pF0": "2K (pF0)",
    "H2K_rF0": "2K (rF0)",
    "H5K_pF0": "5K (pF0)",
    "H5K_rF0": "5K (rF0)",
    "H42Ku_pF0": "H4-2K (pF0)",
    "H42Ku_rF0": "H4-2K (rF0)",
    "H2KH5Ku_pF0": "2K-5K (pF0)",
    "H2KH5Ku_rF0": "2K-5K (rF0)",
    "H42Kc_pF0": "H4*-2K* (pF0)",
    "H42Kc_rF0": "H4*-2K* (rF0)",
    "H2KH5Kc_pF0": "2K*-5K* (pF0)",
    "H2KH5Kc_rF0": "2K*-5K* (rF0)",
    "CPP_pF0": "CPP (pF0)",
    "CPP_rF0": "CPP (rF0)",
    "Energy": "Energy",
    "HNR05_pF0": "HNR05 (pF0)",
    "HNR15_pF0": "HNR15 (pF0)",
    "HNR25_pF0": "HNR25 (pF0)",
    "HNR35_pF0": "HNR35 (pF0)",
    "HNR05_rF0": "HNR05 (rF0)",
    "HNR15_rF0": "HNR15 (rF0)",
    "HNR25_rF0": "HNR25 (rF0)",
    "HNR35_rF0": "HNR35 (rF0)",
    "SHR_pF0": "SHR (pF0)",
    "SHR_rF0": "SHR (rF0)",
    "SpectralSlope_pF0": "Slope (pF0)",
    "SpectralSlope_rF0": "Slope (rF0)",
    "Jitter_Local": "Jitter (Local)",
    "Jitter_RAP": "Jitter (RAP)",
    "Jitter_PPQ5": "Jitter (PPQ5)",
    "Shimmer_Local": "Shimmer (Local)",
    "Shimmer_APQ3": "Shimmer (APQ3)",
    "Shimmer_APQ5": "Shimmer (APQ5)",
    "Shimmer_APQ11": "Shimmer (APQ11)",
    "LipArea": "唇面积",
    "LipWidth": "唇宽度",
    "LipOpen": "唇开度",
    "LipCirc": "唇圆度",
}

CORE_RESULT_FIELD_MAP = {
    "pF0": "f0_praat",
    "rF0": "f0_reaper",
    "pF1": "f1",
    "pF2": "f2",
    "pF3": "f3",
    "pF4": "f4",
    "pB1": "b1",
    "pB2": "b2",
    "pB3": "b3",
    "pB4": "b4",
    "Intensity": "intensity",
}

log = logging.getLogger(__name__)

class AcousticAnalysisService:
    def __init__(self):
        pass

    def analyze_batch(
        self,
        files: List[str],
        input_dir: Path,
        output_dir: Path,
        lip_data_map: Dict[str, Path],
        config: AcousticConfig,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        批量分析文件。
        
        Args:
            files: 相对于 input_dir 的文件路径列表
            input_dir: 输入目录的根路径
            output_dir: 输出目录的根路径
            lip_data_map: 文件名到唇形数据路径的映射
            config: 分析配置
            progress_callback: 进度回调函数 (progress_percentage, current_file_name)
            stop_check: 检查是否需要停止的回调函数，返回 True 则停止
        """
        total = len(files)
        for i, name in enumerate(files):
            if stop_check and stop_check():
                break
                
            if progress_callback:
                progress_callback(int((i / total) * 100), name)
            
            try:
                wav_path = input_dir / name
                lip_path = lip_data_map.get(name)
                tg_path = wav_path.with_suffix(".TextGrid")
                
                # Analyze
                result = self.analyze_file(
                    str(wav_path), 
                    config, 
                    str(lip_path) if lip_path else None,
                    str(tg_path) if tg_path.exists() else None
                )
                
                # Save
                rel_path = Path(name)
                out_path = output_dir / rel_path.with_suffix(".xlsx")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.save_results(result, str(out_path))
                
            except Exception as e:
                log.error(f"Error processing {name}: {e}")
                # Optional: emit error via callback if needed, or just log

    def analyze_file(self, wav_path: str, config: AcousticConfig, lip_pkl_path: Optional[str] = None, textgrid_path: Optional[str] = None) -> AnalysisResult:
        """
        对单个音频文件进行完整的声学参数分析。
        
        Args:
            wav_path: 音频文件路径
            config: 配置参数 (AcousticConfig 对象)
            lip_pkl_path: 唇形数据文件路径 (可选)
            textgrid_path: TextGrid 文件路径 (可选)
            
        Returns:
            包含所有分析结果的 DataFrame
        """
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"文件未找到: {wav_path}")

        # --- 1. 读取音频 ---
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", wavfile.WavFileWarning)
                fs, data = wavfile.read(str(path))
            
            # Convert to float
            if data.dtype == np.int16:
                y = data.astype(np.float64) / 32768.0
            elif data.dtype == np.int32:
                y = data.astype(np.float64) / 2147483648.0
            elif data.dtype == np.uint8:
                y = (data.astype(np.float64) - 128) / 128.0
            else:
                y = data.astype(np.float64)
            
            # Convert to mono
            if y.ndim > 1:
                y = np.mean(y, axis=1)
                
        except Exception as e:
            raise ValueError(f"读取音频失败: {e}")

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
            print(f"Intensity calculation failed: {e}")
            est_len = int(len(y) / (fs * frameshift_ms / 1000.0))
            intensity = np.full(est_len, np.nan)
            silence_mask = np.ones(est_len, dtype=bool)

        # --- 1. 计算 Formants (Praat Burg) ---
        try:
            formant_res = compute_praat_formants(
                path, 
                frameshift_ms, 
                max_formant=config.max_formant,
                num_formants=config.num_formants
            )
        except Exception as e:
            print(f"Formant calculation failed: {e}")
            # Placeholder
            est_len = int(len(y) / (fs * frameshift_ms / 1000))
            formant_res = {f"pF{i}": np.full(est_len, np.nan) for i in range(1, 5)}
            formant_res.update({f"pB{i}": np.full(est_len, np.nan) for i in range(1, 5)})

        # 获取共振峰数据供后续使用
        pF1 = formant_res.get("pF1")
        pF2 = formant_res.get("pF2")
        pF3 = formant_res.get("pF3")
        pF4 = formant_res.get("pF4")
        pB1 = formant_res.get("pB1")
        pB2 = formant_res.get("pB2")
        pB3 = formant_res.get("pB3")
        pB4 = formant_res.get("pB4")

        # --- 3. 计算 F0 (Praat & REAPER) ---
        f0_data = {}
        
        # 3.1 Praat F0
        try:
            pf0 = compute_praat_f0(path, frameshift_ms, min_f0, max_f0, "cc")
            f0_data["pF0"] = pf0
        except Exception as e:
            print(f"Praat F0 failed: {e}")
            f0_data["pF0"] = np.full(len(pF1), np.nan)
            
        # 3.2 REAPER F0 (Optional)
        try:
            if config.use_reaper:
                # REAPER expects seconds, config has ms
                rf0_res = compute_reaper_f0(
                    path, 
                    frameshift_ms / 1000.0, 
                    min_f0, 
                    max_f0,
                    hilbert=config.reaper_hilbert,
                    no_highpass=config.reaper_no_highpass,
                    reaper_bin=config.reaper_bin_path
                )
                f0_data["rF0"] = rf0_res.get("rF0", np.array([]))
            else:
                f0_data["rF0"] = np.full(len(pF1), np.nan)
        except Exception as e:
            # print(f"REAPER F0 failed: {e}")
            f0_data["rF0"] = np.full(len(pF1), np.nan)

        # Ensure lengths match Formants (truncate or pad)
        target_len = len(pF1)
        for k in ["pF0", "rF0"]:
            arr = f0_data[k]
            if len(arr) > target_len:
                f0_data[k] = arr[:target_len]
            elif len(arr) < target_len:
                f0_data[k] = np.pad(arr, (0, target_len - len(arr)), constant_values=np.nan)

        # --- 4. 基于 F0 计算衍生参数 (Harmonics, Amplitudes, Corrections, etc.) ---
        # 我们需要为 pF0 和 rF0 分别计算一套参数
        
        derived_data = {}
        
        for f0_type in ["pF0", "rF0"]:
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
                # print(f"Spectral Batch failed: {e}")
                for k in ["H1", "H2", "H4", "A1", "A2", "A3", "H2K", "H5K"]: 
                    derived_data[f"{k}{suffix}"] = np.full(target_len, np.nan)
                h1=h2=h4=a1=a2=a3=h2k=h5k = np.full(target_len, np.nan)

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
            except:
                pass

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
            except:
                pass

            # 4.6 CPP
            try:
                cpp_val = compute_cpp(y, fs, frameshift_ms, f0, n_periods, voiced_mask)
                derived_data[f"CPP{suffix}"] = cpp_val
            except:
                derived_data[f"CPP{suffix}"] = np.full(target_len, np.nan)
                
            # 4.7 HNR
            try:
                hnr_res = compute_hnr(y, fs, frameshift_ms, f0, n_periods, voiced_mask=voiced_mask)
                # output_text.py expects HNR05_pF0, HNR15_pF0...
                for hk, hv in hnr_res.items():
                    # hk is like HNR05, HNR15
                    derived_data[f"{hk}{suffix}"] = hv[:target_len]
            except:
                pass

            # 4.8 SHR
            try:
                shr_val = compute_shr(y, fs, frameshift_ms, f0, min_f0, max_f0, voiced_mask=voiced_mask)
                derived_data[f"SHR{suffix}"] = shr_val[:target_len]
            except:
                derived_data[f"SHR{suffix}"] = np.full(target_len, np.nan)

            # 4.9 Spectral Slope
            try:
                slope = compute_spectral_slope(y, fs, frameshift_ms, f0, min_pitch=min_f0, voiced_mask=voiced_mask)
                derived_data[f"SpectralSlope{suffix}"] = slope[:target_len]
            except:
                derived_data[f"SpectralSlope{suffix}"] = np.full(target_len, np.nan)

            # 4.10 SOE (Strength of Excitation)
            try:
                # Use compute_soe which uses ZFF
                # compute_soe returns (soe_array, epoch_indices)
                soe_val, _ = compute_soe(y, fs, frameshift_ms, f0, target_len)
                derived_data[f"SOE{suffix}"] = soe_val
            except Exception as e:
                # print(f"SOE failed: {e}")
                derived_data[f"SOE{suffix}"] = np.full(target_len, np.nan)

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
            js_res = compute_jitter_shimmer(y, fs, frameshift_ms, js_win, voiced_mask=(f0_data["pF0"] > 0), min_f0=min_f0, max_f0=max_f0)
            for k, v in js_res.items():
                derived_data[k] = v[:target_len]
        except Exception as e:
            # print(f"Jitter/Shimmer failed: {e}")
            for k in ["Jitter_Local", "Jitter_RAP", "Jitter_PPQ5", "Shimmer_Local", "Shimmer_APQ3", "Shimmer_APQ5", "Shimmer_APQ11"]:
                derived_data[k] = np.full(target_len, np.nan)
            
        # CPP (Generic) - usually copy CPP_pF0
        if "CPP_pF0" in derived_data:
            pass
            # derived_data["CPP"] = derived_data["CPP_pF0"] # Removed as per user request (duplicate)

        # --- 6. Lip Data ---
        lip_data = {}
        if lip_pkl_path:
            try:
                # Time vector for interpolation
                times = np.arange(target_len) * frameshift_ms / 1000.0
                # Use smooth_win=1 to disable internal smoothing, we apply global smoothing later
                lip_res = read_lip_data(lip_pkl_path, times, smooth_win=1)
                lip_data.update(lip_res)
            except Exception as e:
                print(f"Lip data processing failed: {e}")

        # --- 7. TextGrid Data ---
        textgrid_data = {}
        if textgrid_path:
            try:
                tg = parse_textgrid(textgrid_path)
                if tg and tg.tiers:
                    times = np.arange(target_len) * frameshift_ms / 1000.0
                    for tier in tg.tiers:
                        # Create a column for each tier
                        col_name = f"text_{tier.name}"
                        labels = np.full(target_len, "", dtype=object)
                        
                        # Optimization: Sort intervals by xmin to allow faster search? 
                        # Intervals are usually sorted.
                        # Simple implementation: For each time point, find interval.
                        # Faster implementation: Iterate intervals and fill range in array.
                        
                        for interval in tier.intervals:
                            # Find indices corresponding to this interval
                            # t >= xmin and t < xmax
                            start_idx = int(np.ceil(interval.xmin * 1000.0 / frameshift_ms))
                            end_idx = int(np.ceil(interval.xmax * 1000.0 / frameshift_ms))
                            
                            # Clip to valid range
                            start_idx = max(0, start_idx)
                            end_idx = min(target_len, end_idx)
                            
                            if end_idx > start_idx:
                                labels[start_idx:end_idx] = interval.text
                                
                        textgrid_data[col_name] = labels
            except Exception as e:
                print(f"TextGrid processing failed: {e}")

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
            time_axis=np.arange(target_len) * frameshift_ms / 1000.0,
            
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
            def smooth_arr(arr):
                if arr is None or len(arr) == 0: return arr
                return pd.Series(arr).rolling(window=win_size, center=True, min_periods=1).mean().values

            if result.f0_praat is not None: result.f0_praat = smooth_arr(result.f0_praat)
            if result.f0_reaper is not None: result.f0_reaper = smooth_arr(result.f0_reaper)
            
            for attr in ['f1', 'f2', 'f3', 'f4', 'b1', 'b2', 'b3', 'b4']:
                val = getattr(result, attr)
                if val is not None:
                    setattr(result, attr, smooth_arr(val))
                    
            for k, v in result.parameters.items():
                if np.issubdtype(v.dtype, np.number):
                     result.parameters[k] = smooth_arr(v)

            # Smooth lip data
            if result.lip_data:
                for k, v in result.lip_data.items():
                    if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                        result.lip_data[k] = smooth_arr(v)
        
        return self._apply_selected_parameters(result, config.selected_parameter_keys)

    def _apply_selected_parameters(self, result: AnalysisResult, selected_keys: Optional[List[str]]) -> AnalysisResult:
        if not selected_keys:
            return result
        selected = set(selected_keys)
        for key, attr in CORE_RESULT_FIELD_MAP.items():
            if key not in selected and hasattr(result, attr):
                setattr(result, attr, None)
        result.parameters = {k: v for k, v in result.parameters.items() if k in selected}
        result.lip_data = {k: v for k, v in result.lip_data.items() if k in selected}
        return result

    def save_results(self, result: Union[pd.DataFrame, AnalysisResult], output_path: str):
        p = Path(output_path)
        # Force xlsx if possible, but respect user choice if provided
        if p.suffix.lower() != ".xlsx":
            p = p.with_suffix(".xlsx")
            
        if isinstance(result, AnalysisResult):
            df = result.to_dataframe()
        else:
            df = result
            
        # Apply renaming for display purposes
        df_to_save = df.copy()
        
        # Rename columns using PARAMETER_MAPPING
        # Note: We do NOT rename text_ columns here as they are already prefixed
        df_to_save.rename(columns=PARAMETER_MAPPING, inplace=True)
            
        data_dict = df_to_save.to_dict(orient='list')
        try:
            save_excel(p, data_dict)
            save_fast_parameter_db(p, df_to_save)
        except Exception as e:
            log.error(f"Failed to save results: {e}")
