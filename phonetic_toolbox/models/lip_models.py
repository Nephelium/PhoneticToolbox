from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LipLaunchResult:
    success: bool
    message: str
    target_path: str = ""
    working_directory: str = ""
    command: list[str] = field(default_factory=list)
    process_id: Optional[int] = None
