# Existing unified host, source adapters and a project-only legacy runtime.
$ErrorActionPreference = 'Stop'
$m05Root = Split-Path -Parent $PSScriptRoot
$m05Python = Join-Path $m05Root '.venv/m05/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $m05Python)) { throw 'M05 project runtime is missing; see docs/manual/lip-extraction.md.' }
$m05PreviousPath = $env:PYTHONPATH
$m05PreviousPython = $env:PTB_M05_PYTHON
try {
    $m05Sources = @('packages/phonetic_core/src', 'backend/src', 'desktop/src') | ForEach-Object { Join-Path $m05Root $_ }
    $env:PYTHONPATH = ($m05Sources -join ';') + $(if ($m05PreviousPath) { ';' + $m05PreviousPath } else { '' })
    $env:PTB_M05_PYTHON = $m05Python
    & (Join-Path $PSScriptRoot 'Start-Research-Workbench.ps1')
} finally {
    $env:PYTHONPATH = $m05PreviousPath
    $env:PTB_M05_PYTHON = $m05PreviousPython
}
