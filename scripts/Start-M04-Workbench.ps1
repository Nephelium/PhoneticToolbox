# M04 source repair entry. Reuse the unified host and existing task store.
$ErrorActionPreference = 'Stop'
$m04Root = Split-Path -Parent $PSScriptRoot
$m04PreviousPath = $env:PYTHONPATH
try {
    $m04Sources = @('packages/phonetic_core/src', 'backend/src', 'desktop/src') | ForEach-Object { Join-Path $m04Root $_ }
    $env:PYTHONPATH = ($m04Sources -join ';') + $(if ($m04PreviousPath) { ';' + $m04PreviousPath } else { '' })
    & (Join-Path $PSScriptRoot 'Start-Research-Workbench.ps1')
} finally {
    $env:PYTHONPATH = $m04PreviousPath
}
