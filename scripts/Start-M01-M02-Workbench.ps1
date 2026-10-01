# M01/M02 current source on the existing shared local host. No install or DDL.
$ErrorActionPreference = 'Stop'
$parameterRoot = Split-Path -Parent $PSScriptRoot
$parameterPreviousSources = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (@('packages/phonetic_core/src','backend/src','desktop/src') | ForEach-Object { Join-Path $parameterRoot $_ }) -join ';'
    & (Join-Path $PSScriptRoot 'Start-Research-Workbench.ps1')
} finally {
    $env:PYTHONPATH = $parameterPreviousSources
}
