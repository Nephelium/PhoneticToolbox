param(
    [string]$Rules,
    [string]$Python,
    [string]$Manifest,
    [ValidatePattern('^[a-zA-Z0-9-]+\.json$')][string]$ReportName,
    [switch]$Apply
)
# Default is review only; ownership/evidence, hashes and process checks remain mandatory.
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if (-not $Rules) { $Rules = Join-Path $root 'output\maintenance\test-temporaries-rules.json' }
if (-not $Python) { $Python = Join-Path $root '.venv\m14\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Rules -PathType Leaf)) { throw 'Create and review output/maintenance/test-temporaries-rules.json first.' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'The project Python is missing; supply -Python with the source-workbench interpreter.' }
if (-not $Manifest) { $Manifest = Join-Path $root 'output\maintenance\test-temporaries-plan.json' }
& $Python -B -X utf8 (Join-Path $PSScriptRoot 'prepare_test_cleanup.py') --rules $Rules --out $Manifest
if ($LASTEXITCODE -ne 0) { throw 'Manifest preparation failed; no deletion was attempted.' }
if (-not $ReportName) { $ReportName = if ($Apply) { 'test-temporaries-results.json' } else { 'test-temporaries-review.json' } }
& (Join-Path $PSScriptRoot 'cleanup_workspace.ps1') -Manifest $Manifest -Apply:$Apply -ReportName $ReportName
exit $LASTEXITCODE
