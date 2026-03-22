from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np

# --- Constants ---
DEFAULT_ROI_START = 0.0
DEFAULT_ROI_DURATION = 0.5

@dataclass
class AcousticConfig:
    """声学参数分析配置"""
    min_f0: float = 60.0
    max_f0: float = 880.0
    frameshift_ms: float = 5.0
    windowsize_ms: float = 40.0
    
    # REAPER specific
    use_reaper: bool = True
    reaper_hilbert: bool = True
    reaper_no_highpass: bool = False
    reaper_bin_path: str = "phonetic_toolbox/core/acoustic/reaper.exe"
    
    # Formant specific
    max_formant: float = 6000.0
    num_formants: int = 5
    
    # Silence/Voicing
    silence_threshold: float = 0.03  # Energy threshold for silence (relative to max intensity)
    energy_window_ms: float = 40.0
    
    # Analysis specific
    n_periods: int = 3
    smooth_win_size: int = 10
    lip_smooth_win_size: int = 0
    only_voiced: bool = True  # If True, unvoiced frames will be set to NaN based on ZCR/Energy
    selected_parameter_keys: Optional[List[str]] = None

@dataclass
class PitchManipulationConfig:
    """变调配置"""
    time_step: float = 0.01
    min_pitch: float = 75.0
    max_pitch: float = 600.0
    
    # Batch processing
    start_mode: str = "order" # full | order | reverse | constant
    end_mode: str = "order"
    offset_mode: bool = False

@dataclass
class EGGConfig:
    """EGG 分析配置"""
    peak_prominence: float = 0.01
    valley_prominence: float = 0.01
    auto_prominence: bool = True
    min_auto_prominence: float = 0.01
    
    # Filter
    highpass_cutoff: float = 25.0
    lowpass_cutoff: float = 1000.0
    
    # GCI/GOI Method
    gci_method: str = "slope" # slope | scale
    goi_method: str = "slope" # slope | scale
    criterion_level: float = 0.25 # For scale method
    
    # Spectrogram
    spec_window_ms: float = 20.0
    spec_vmin: float = -70.0
    spec_vmax: float = -10.0
    
    # Inverse Filtering
    if_order_heuristic_add: int = 6 # fs/1000 + 6

import pandas as pd

@dataclass
class AnalysisResult:
    """分析结果数据模型"""
    time_axis: np.ndarray
    
    # F0
    f0_praat: np.ndarray
    f0_reaper: Optional[np.ndarray] = None
    
    # Formants
    f1: Optional[np.ndarray] = None
    f2: Optional[np.ndarray] = None
    f3: Optional[np.ndarray] = None
    f4: Optional[np.ndarray] = None
    b1: Optional[np.ndarray] = None
    b2: Optional[np.ndarray] = None
    b3: Optional[np.ndarray] = None
    b4: Optional[np.ndarray] = None
    
    # Intensity
    intensity: Optional[np.ndarray] = None
    
    # Spectral & Voice Quality Parameters (stored in extras to allow flexibility)
    # e.g., H1, H2, H4, A1, A2, A3, H2K, H5K, CPP, HNR, SHR, SoE, Jitter, Shimmer
    # and their corrected versions
    parameters: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # External Data
    lip_data: Dict[str, np.ndarray] = field(default_factory=dict)
    textgrid_data: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Metadata
    sampling_rate: int = 16000
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert result to pandas DataFrame"""
        data = {
            "Time_s": self.time_axis
        }
        
        # Add core parameters
        if self.f0_praat is not None: data["pF0"] = self.f0_praat
        if self.f0_reaper is not None: data["rF0"] = self.f0_reaper
        
        if self.f1 is not None: data["pF1"] = self.f1
        if self.f2 is not None: data["pF2"] = self.f2
        if self.f3 is not None: data["pF3"] = self.f3
        if self.f4 is not None: data["pF4"] = self.f4
        
        if self.b1 is not None: data["pB1"] = self.b1
        if self.b2 is not None: data["pB2"] = self.b2
        if self.b3 is not None: data["pB3"] = self.b3
        if self.b4 is not None: data["pB4"] = self.b4
        
        if self.intensity is not None: data["Intensity"] = self.intensity
        
        # Add dynamic parameters
        data.update(self.parameters)
        data.update(self.lip_data)
        data.update(self.textgrid_data)
        
        # Ensure all arrays have same length as time_axis
        target_len = len(self.time_axis)
        for k, v in data.items():
            if len(v) > target_len:
                data[k] = v[:target_len]
            elif len(v) < target_len:
                # Pad with NaN (or empty string for object/text arrays if needed, but np.pad handles numeric)
                if v.dtype.kind in {'U', 'S', 'O'}: # String/Object
                     # Manual padding for object arrays
                     new_v = np.full(target_len, "", dtype=object)
                     new_v[:len(v)] = v
                     data[k] = new_v
                else:
                    data[k] = np.pad(v, (0, target_len - len(v)), constant_values=np.nan)

        return pd.DataFrame(data)
