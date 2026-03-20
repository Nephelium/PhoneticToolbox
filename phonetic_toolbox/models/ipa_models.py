from dataclasses import dataclass, field


@dataclass
class IPATransLaunchResult:
    success: bool
    message: str
    script_path: str = ""
    html_path: str = ""
    working_directory: str = ""
    command: list[str] = field(default_factory=list)
    generator_output: str = ""
