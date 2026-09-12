"""M03 local migration of v2 EGG behavior. Source: PENDING-EGG; provenance unresolved.
See NOTICE.txt. Pure numerical compatibility layer, no file/device/GUI access.
"""
from dataclasses import dataclass

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
