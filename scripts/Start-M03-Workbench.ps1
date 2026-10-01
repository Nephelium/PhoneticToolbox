# Current unified source host plus the existing locked M03 runtime. No install/DDL.
$ErrorActionPreference = 'Stop'
$m03Root = Split-Path -Parent $PSScriptRoot
$m03PreviousSources = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (@('packages/phonetic_core/src','backend/src','desktop/src') | ForEach-Object { Join-Path $m03Root $_ }) -join ';'
    & (Join-Path $PSScriptRoot 'Start-Research-Workbench.ps1')
} finally {
    $env:PYTHONPATH = $m03PreviousSources
}
