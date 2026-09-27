# M06 source entry into the shared production workbench. No install/build/DDL.
$ErrorActionPreference = 'Stop'
$m06Root = Split-Path -Parent $PSScriptRoot
$m06Python = Join-Path $m06Root '.venv/m09-ui/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $m06Python)) { throw 'M06 reviewed Windows runtime is missing.' }
$m06PreviousPath = $env:PYTHONPATH
$m06PreviousEgg = $env:PTB_EGG_PYTHON
try {
    $env:PYTHONPATH = (@('packages/phonetic_core/src','backend/src','desktop/src') | ForEach-Object { Join-Path $m06Root $_ }) -join ';'
    $m06Egg = Join-Path $m06Root '.venv/m03-compatible/python.exe'
    if (Test-Path -LiteralPath $m06Egg) { $env:PTB_EGG_PYTHON = $m06Egg }
    & $m06Python -B -X utf8 (Join-Path $PSScriptRoot 'start_m01_workbench.py')
    if ($LASTEXITCODE -ne 0) { throw 'M06 workbench stopped. See docs/testing/m06-report.md.' }
} finally {
    $env:PYTHONPATH = $m06PreviousPath
    $env:PTB_EGG_PYTHON = $m06PreviousEgg
}
