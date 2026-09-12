# M03 scientific validation runtime only. Does not start Qt, install or migrate data.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArguments)
$ErrorActionPreference = 'Stop'
$m03Root = Split-Path -Parent $PSScriptRoot
$m03Prefix = Join-Path $m03Root '.venv/m03-compatible'
$m03Python = Join-Path $m03Prefix 'python.exe'
if (-not (Test-Path -LiteralPath $m03Python)) { throw 'M03 compatible scientific runtime is missing.' }
$m03PreviousPath = $env:PATH
try {
    $env:PATH = "$m03Prefix;$m03Prefix\Library\bin;$m03Prefix\Scripts;$m03PreviousPath"
    & $m03Python @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "M03 Python exited with code $LASTEXITCODE." }
} finally {
    $env:PATH = $m03PreviousPath
}
