"""M03 local migration of v2 EGG behavior. Source: PENDING-EGG; provenance unresolved.
See NOTICE.txt. Pure numerical compatibility layer, no file/device/GUI access.
"""
from dataclasses import dataclass
from .errors import EggError, finite_number

@dataclass(frozen=True)
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

    def __post_init__(self):
        for name in ('peak_prominence', 'valley_prominence', 'min_auto_prominence',
                     'highpass_cutoff', 'lowpass_cutoff', 'spec_window_ms'):
            value = getattr(self, name)
            if not finite_number(value) or value < 0 or (name in ('highpass_cutoff', 'lowpass_cutoff', 'spec_window_ms') and value == 0):
                raise EggError('invalid_config_' + name)
        if self.gci_method not in ('slope', 'scale') or self.goi_method not in ('slope', 'scale'):
            raise EggError('invalid_event_method')
        if type(self.auto_prominence) is not bool:
            raise EggError('invalid_auto_prominence')
        if not finite_number(self.criterion_level) or self.criterion_level != .25:
            raise EggError('legacy_criterion_is_fixed_025')
        if type(self.if_order_heuristic_add) is not int or self.if_order_heuristic_add != 6:
            raise EggError('legacy_inverse_heuristic_is_fixed_6')
        if not finite_number(self.spec_vmin) or not finite_number(self.spec_vmax) or self.spec_vmin >= self.spec_vmax:
            raise EggError('invalid_spectrogram_range')
        if self.highpass_cutoff >= self.lowpass_cutoff:
            raise EggError('filter_cutoff')

    @classmethod
    def for_workbench(cls, **overrides):
        """Actual v2 single-file and batch UI defaults; the bare service default differs."""
        return cls(**{'goi_method': 'scale', **overrides})
