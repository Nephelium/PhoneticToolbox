# Unified M11 source entry, no install/build/migration or global environment edit.
param([string]$ComponentRoot = '')
$ErrorActionPreference = 'Stop'
$m11Root = Split-Path -Parent $PSScriptRoot
$m11Python = Join-Path $m11Root '.venv/v3-dev/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $m11Python)) { throw 'The verified project v3-dev runtime is missing.' }
$m11PreviousPath = $env:PYTHONPATH
$m11PreviousComponents = $env:PTB_M11_COMPONENT_ROOT
try {
    $env:PYTHONPATH = (@('packages/phonetic_core/src','backend/src','desktop/src') | ForEach-Object { Join-Path $m11Root $_ }) -join ';'
    if ($ComponentRoot) { $env:PTB_M11_COMPONENT_ROOT = [IO.Path]::GetFullPath($ComponentRoot) }
    & $m11Python -B -X utf8 (Join-Path $PSScriptRoot 'start_m01_workbench.py')
    if ($LASTEXITCODE -ne 0) { throw 'The workbench stopped. See the M11 validation report.' }
} finally {
    $env:PYTHONPATH = $m11PreviousPath
    $env:PTB_M11_COMPONENT_ROOT = $m11PreviousComponents
}
