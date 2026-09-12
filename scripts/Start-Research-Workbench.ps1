# M01/M02/M09 shared local development entry. No build, update or migration.
$ErrorActionPreference = 'Stop'
$researchRoot = Split-Path -Parent $PSScriptRoot
$researchPython = Join-Path $researchRoot '.venv/m09-ui/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $researchPython)) { throw 'M02/M09 project runtime is missing.' }
& $researchPython -X utf8 (Join-Path $PSScriptRoot 'start_m01_workbench.py')
if ($LASTEXITCODE -ne 0) { throw 'The local workbench stopped. See the M02/M09 verification report.' }
