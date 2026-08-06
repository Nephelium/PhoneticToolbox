from dataclasses import dataclass


@dataclass
class ArticulatorySynthLaunchResult:
    success: bool
    message: str
    html_path: str = ""
    working_directory: str = ""
