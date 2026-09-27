# M07 source entry into the shared production workbench. No install/build/DDL.
$ErrorActionPreference = 'Stop'
$m07Root = Split-Path -Parent $PSScriptRoot
$m07Python = Join-Path $m07Root '.venv/m09-ui/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $m07Python)) { throw 'M07 reviewed Windows runtime is missing.' }
$m07PreviousPath = $env:PYTHONPATH
$m07PreviousEgg = $env:PTB_EGG_PYTHON
try {
    $env:PYTHONPATH = (@('packages/phonetic_core/src','backend/src','desktop/src') | ForEach-Object { Join-Path $m07Root $_ }) -join ';'
    $m07Egg = Join-Path $m07Root '.venv/m03-compatible/python.exe'
    if (Test-Path -LiteralPath $m07Egg) { $env:PTB_EGG_PYTHON = $m07Egg }
    & $m07Python -B -X utf8 (Join-Path $PSScriptRoot 'start_m01_workbench.py')
    if ($LASTEXITCODE -ne 0) { throw 'M07 workbench stopped. See docs/testing/m07-report.md.' }
} finally {
    $env:PYTHONPATH = $m07PreviousPath
    $env:PTB_EGG_PYTHON = $m07PreviousEgg
}
