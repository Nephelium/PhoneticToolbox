# Local development entry. Uses existing reviewed schema and installed wheels.
$ErrorActionPreference = 'Stop'
$m01Root = Split-Path -Parent $PSScriptRoot
$m01Python = Join-Path $m01Root '.venv/m01-ui/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $m01Python)) { throw 'M01 project runtime is missing.' }
& $m01Python -X utf8 (Join-Path $PSScriptRoot 'start_m01_workbench.py')
if ($LASTEXITCODE -ne 0) { throw 'The local workbench stopped. Check the M01 validation report before retrying.' }
