from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class EGGAnalysisResult:
    """EGG 分析结果"""
    # Time vectors
    time_vector: np.ndarray # Original time vector for signals
    
    # Signals
    egg_signal_raw: np.ndarray
    egg_signal_processed: np.ndarray # Filtered/Denoised
    audio_signal: np.ndarray
    
    # Events (Global)
    gci_times: List[float] = field(default_factory=list)
    goi_times: List[float] = field(default_factory=list)
    peak_times: List[float] = field(default_factory=list)
    
    # Derived Parameters (Global/Segment)
    cq_times: Optional[np.ndarray] = None
    cq_values: Optional[np.ndarray] = None
    sq_values: Optional[np.ndarray] = None
    
    # F0 Data
    gci_f0_times: Optional[np.ndarray] = None
    gci_f0_values: Optional[np.ndarray] = None
    audio_f0_times: Optional[np.ndarray] = None
    audio_f0_values: Optional[np.ndarray] = None
    
    # Metadata
    fs: int = 44100
    file_duration: float = 0.0
    
    # Glottal Movement
    glottal_movement_events: List[Tuple[float, str]] = field(default_factory=list) # List of (time, type)
