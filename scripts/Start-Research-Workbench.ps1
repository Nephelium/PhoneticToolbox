# One source entry for every module. No build, install or migration.
param(
    [ValidateSet('home','M01','M02','M03','M04','M05','M06','M07','M08','M09','M10','M11','M12','M13','M14','M15','M16','M17','M18')]
    [string]$Module = 'home',
    [string]$ComponentRoot = '',
    [switch]$CheckOnly,
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$researchRoot = Split-Path -Parent $PSScriptRoot
$researchPython = Join-Path $researchRoot '.venv/m14/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $researchPython)) { throw 'The shared project host runtime is missing: .venv/m14.' }
$researchArguments = @('-E', '-s', '-B', '-X', 'utf8', (Join-Path $PSScriptRoot 'workbench_source.py'), '--module', $Module)
if ($ComponentRoot) { $researchArguments += @('--component-root', [IO.Path]::GetFullPath($ComponentRoot)) }
if ($CheckOnly) { $researchArguments += '--check-only' }
if ($PrepareOnly) { $researchArguments += '--prepare-only' }
& $researchPython @researchArguments
if ($LASTEXITCODE -ne 0) { throw 'The source workbench failed. See the error above; no installed-code fallback is used.' }
