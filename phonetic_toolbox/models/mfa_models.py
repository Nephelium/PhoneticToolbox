from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MFAAlignmentConfig:
    beam: int = 10
    retry_beam: int = 40


@dataclass
class MFAAutoAlignmentLaunchResult:
    success: bool
    message: str
    project_dir: str = ""
    batch_file: str = ""
    working_directory: str = ""
    command: list[str] = field(default_factory=list)
    process_id: Optional[int] = None


@dataclass
class MFAAlignmentRunResult:
    success: bool
    message: str
    output_path: str = ""
    detail: str = ""
