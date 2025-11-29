from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import platform


@dataclass
class AppState:
    """全局状态，复刻 MATLAB vs_Initialize 的默认变量。"""

    dirdelimiter: str = field(default_factory=lambda: "\\" if platform.system() == "Windows" else "/")

    wavdir: str = field(default_factory=lambda: f".{AppState._delim()}")
    matdir: str = field(default_factory=lambda: f".{AppState._delim()}")

    windowsize: int = 25
    frameshift: int = 2
    preemphasis: float = 0.96
    lpcOrder: int = 12
    maxstrF0: int = 500
    minstrF0: int = 40
    maxF0: int = 500
    minF0: int = 40
    maxstrdur: int = 10
    tbuffer: int = 25
    F0OtherEnable: int = 0
    F0OtherOffset: int = 0
    F0OtherCommand: str = ""
    FormantsOtherEnable: int = 0
    FormantsOtherOffset: int = 0
    FormantsOtherCommand: str = ""
    TextgridIgnoreList: str = '"", " ", "SIL"'
    TextgridTierNumber: int = 1
    frame_precision: int = 1

    F0Praatmax: int = 500
    F0Praatmin: int = 40
    F0PraatVoiceThreshold: float = 0.45
    F0PraatOctiveJumpCost: float = 0.35
    F0PraatSilenceThreshold: float = 0.03
    F0PraatOctaveCost: float = 0.01
    F0PraatOctaveJumpCost: float = 0.35
    F0PraatVoicedUnvoicedCost: float = 0.14
    F0PraatKillOctaveJumps: int = 0
    F0PraatSmooth: int = 0
    F0PraatSmoothingBandwidth: int = 5
    F0PraatInterpolate: int = 0
    F0Praatmethod: str = "cc"
    FormantsPraatMaxFormantFreq: int = 6000
    FormantsPraatNumFormants: int = 4

    F0ReaperFrameIntervalSec: float = 0.002
    F0ReaperMinF0: int = 20
    F0ReaperMaxF0: int = 500
    F0ReaperHilbert: int = 1
    F0ReaperNoHighpass: int = 0
    F0ReaperBin: str = field(default_factory=lambda: str(Path.cwd() / "reaper.exe"))

    BandwidthMethods: tuple[str, str] = ("Use formula values", "Use estimated values")
    BandwidthMethod: str = "Use formula values"

    recursedir: int = 0
    linkmatdir: int = 1
    linkwavdir: int = 1

    Nperiods: int = 3
    Nperiods_EC: int = 5

    SHRmax: int = 500
    SHRmin: int = 40
    SHRThreshold: float = 0.4

    EGGheaders: str = "CQ, CQ_H, CQ_PM, CQ_HT, peak_Vel, peak_Vel_Time, min_Vel, min_Vel_Time, SQ2-SQ1, SQ4-SQ3, ratio"
    EGGtimelabel: str = "Frame"

    F0algorithm: str = "F0 (REAPER)"
    FMTalgorithm: str = "Formants (Snack)"

    # 参数估计（PE）
    PE_savematwithwav: int = 1
    PE_processwith16k: int = 1
    PE_useTextgrid: int = 1
    PE_showwaveforms: int = 0
    PE_params: list[str] = field(default_factory=list)

    # 参数显示（PD）
    PD_wavdir: str = field(default_factory=lambda: f".{AppState._delim()}")
    PD_matdir: str = field(default_factory=lambda: f".{AppState._delim()}")
    PD_paramselection: list[int] = field(default_factory=list)
    PD_plottype: tuple[str, ...] = ("b", "r", "g", "k", "c", "b:", "r:", "g:", "k:", "c:", "b--", "r--", "g--", "k--", "c--")
    PD_maxplots: int = 15
    PD_param_combos: dict[str, list[str]] = field(default_factory=dict)

    # 能量阈值（按帧能量相对最大值的比例），用于无声段掩蔽
    voicing_energy_threshold_ratio: float = 0.01
    voicing_energy_window_ms: int = 25

    # 输出到文本（OT）
    OT_selectedParams: list[str] = field(default_factory=list)
    OT_matdir: str = field(default_factory=lambda: f".{AppState._delim()}")
    OT_includesubdir: int = 1
    OT_Textgriddir: str = field(default_factory=lambda: f".{AppState._delim()}")
    OT_includeEGG: int = 0
    OT_EGGdir: str = field(default_factory=lambda: f".{AppState._delim()}")
    OT_outputdir: str = field(default_factory=lambda: f".{AppState._delim()}")
    OT_includeTextgridLabels: int = 1
    OT_columndelimiter: int = 1
    OT_noSegments: int = 1
    OT_useSegments: int = 0
    OT_numSegments: int = 9
    OT_singleFile: int = 1
    OT_multipleFiles: int = 0
    OT_outputAlgorithmMetadata: int = 0

    OT_singleFilename: str = "output.txt"
    OT_F0CPPEfilename: str = "F0_CPP_E_HNR.txt"
    OT_Formantsfilename: str = "Formants.txt"
    OT_Hx_Axfilename: str = "HA.txt"
    OT_HxHxfilename: str = "HxHx.txt"
    OT_HxAxfilename: str = "HxAx.txt"
    OT_Epochfilename: str = "Epoch.txt"
    OT_EGGfilename: str = "EGG.txt"

    OT_Single: str = field(init=False)
    OT_F0CPPE: str = field(init=False)
    OT_Formants: str = field(init=False)
    OT_HA: str = field(init=False)
    OT_HxHx: str = field(init=False)
    OT_HxAx: str = field(init=False)
    OT_Epoch: str = field(init=False)
    OT_EGG: str = field(init=False)

    # Manual Data（MD）
    MD_wavdir: str = field(default_factory=lambda: f".{AppState._delim()}")
    MD_matdir: str = field(default_factory=lambda: f".{AppState._delim()}")
    MD_offset: int = 0
    MD_resample: int = 0
    MD_matwithwav: int = 1


    # Outputs
    O_smoothwinsize: int = 0

    # 输入（wav）
    I_searchstring: str = "*.wav"

    def __post_init__(self) -> None:
        self.OT_Single = f"{self.OT_outputdir}{self.dirdelimiter}{self.OT_singleFilename}"
        self.OT_F0CPPE = f"{self.OT_outputdir}{self.dirdelimiter}{self.OT_F0CPPEfilename}"
        self.OT_Formants = f"{self.OT_outputdir}{self.dirdelimiter}{self.OT_Formantsfilename}"
        self.OT_HA = f"{self.OT_outputdir}{self.dirdelimiter}{self.OT_Hx_Axfilename}"
        self.OT_HxHx = f"{self.OT_outputdir}{self.dirdelimiter}{self.OT_HxHxfilename}"
        self.OT_HxAx = f"{self.OT_outputdir}{self.dirdelimiter}{self.OT_HxAxfilename}"
        self.OT_Epoch = f"{self.OT_outputdir}{self.dirdelimiter}{self.OT_Epochfilename}"
        self.OT_EGG = f"{self.OT_outputdir}{self.dirdelimiter}{self.OT_EGGfilename}"

    @staticmethod
    def _delim() -> str:
        return "\\" if platform.system() == "Windows" else "/"
