from dataclasses import dataclass


@dataclass
class PerceptionLaunchResult:
    success: bool
    message: str
    html_path: str = ""
    working_directory: str = ""
