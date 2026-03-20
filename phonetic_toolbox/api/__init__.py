from phonetic_toolbox.services.acoustic_service import AcousticAnalysisService
from phonetic_toolbox.services.settings_service import SettingsService
from phonetic_toolbox.services.lip_service import LipExtractionService
from phonetic_toolbox.services.ipa_trans_service import IPATransService
from phonetic_toolbox.services.perception_service import (
    PerceptionExperimentService,
)
from phonetic_toolbox.services.mfa_alignment_service import (
    MFAAutoAlignmentService,
)
from phonetic_toolbox.services.lpc_service import LPCSpectrumService
from phonetic_toolbox.models.config import AcousticConfig, AnalysisResult
from phonetic_toolbox.models.lip_models import LipLaunchResult
from phonetic_toolbox.models.ipa_models import IPATransLaunchResult
from phonetic_toolbox.models.perception_models import PerceptionLaunchResult
from phonetic_toolbox.models.mfa_models import (
    MFAAlignmentRunResult,
    MFAAutoAlignmentLaunchResult,
)


def launch_lip_extraction(
    load_existing: bool = False, project_dir: str | None = None
) -> LipLaunchResult:
    service = LipExtractionService(project_dir=project_dir)
    return service.launch(load_existing=load_existing)


def launch_ipa_trans(project_dir: str | None = None) -> IPATransLaunchResult:
    service = IPATransService(project_dir=project_dir)
    return service.launch()


def launch_perception_experiment(
    project_dir: str | None = None,
) -> PerceptionLaunchResult:
    service = PerceptionExperimentService(project_dir=project_dir)
    return service.launch()


def launch_mfa_auto_alignment(
    project_dir: str | None = None,
) -> MFAAutoAlignmentLaunchResult:
    service = MFAAutoAlignmentService(project_dir=project_dir)
    return service.launch()


__all__ = [
    "AcousticAnalysisService",
    "SettingsService",
    "LipExtractionService",
    "AcousticConfig",
    "AnalysisResult",
    "IPATransService",
    "PerceptionExperimentService",
    "MFAAutoAlignmentService",
    "LPCSpectrumService",
    "LipLaunchResult",
    "IPATransLaunchResult",
    "PerceptionLaunchResult",
    "MFAAutoAlignmentLaunchResult",
    "MFAAlignmentRunResult",
    "launch_lip_extraction",
    "launch_ipa_trans",
    "launch_perception_experiment",
    "launch_mfa_auto_alignment",
]
