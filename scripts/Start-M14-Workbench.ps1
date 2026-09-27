# M14 source entry into the existing unified workbench. No install/build/migration.
$ErrorActionPreference = 'Stop'
$m14Root = Split-Path -Parent $PSScriptRoot
$m14PreviousPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (@('packages/phonetic_core/src','backend/src','desktop/src') | ForEach-Object { Join-Path $m14Root $_ }) -join ';'
    & (Join-Path $m14Root '.venv/m14/Scripts/python.exe') -B -X utf8 (Join-Path $PSScriptRoot 'start_m01_workbench.py')
    if ($LASTEXITCODE -ne 0) { throw 'M14 workbench stopped. See M14 validation report.' }
} finally { $env:PYTHONPATH = $m14PreviousPath }
