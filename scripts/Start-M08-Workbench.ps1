# M08 development entry: the existing unified host with current V3 source adapters.
# No package installation, database migration, build, or separate HTTP server.
$ErrorActionPreference = 'Stop'
$m08Root = Split-Path -Parent $PSScriptRoot
$m08PreviousPath = $env:PYTHONPATH
try {
    $m08Sources = @('packages/phonetic_core/src', 'backend/src', 'desktop/src') | ForEach-Object { Join-Path $m08Root $_ }
    $env:PYTHONPATH = ($m08Sources -join ';') + $(if ($m08PreviousPath) { ';' + $m08PreviousPath } else { '' })
    & (Join-Path $PSScriptRoot 'Start-Research-Workbench.ps1')
} finally {
    $env:PYTHONPATH = $m08PreviousPath
}
