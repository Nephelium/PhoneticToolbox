from dataclasses import dataclass


@dataclass(frozen=True)
class VocalTractLaunchResult:
    success: bool
    message: str
    url: str = ''
    process_id: int | None = None
