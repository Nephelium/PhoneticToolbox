param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$feedbackProjectRoot = Split-Path -Parent $PSScriptRoot
$feedbackPython = Join-Path $feedbackProjectRoot '.venv/m14/Scripts/pythonw.exe'
$feedbackEntry = Join-Path $PSScriptRoot 'start_feedback_manager.py'
if (-not (Test-Path -LiteralPath $feedbackPython)) { throw 'The existing project Qt runtime is unavailable.' }
if (-not (Test-Path -LiteralPath (Join-Path $feedbackProjectRoot 'frontend/dist/feedback-manager.html'))) { throw 'Build the shared frontend first.' }
if ($CheckOnly) { Write-Output 'Feedback manager entry and runtime are available.'; exit 0 }
Start-Process -FilePath $feedbackPython -ArgumentList @('-E','-s','-B','-X','utf8',$feedbackEntry) -WorkingDirectory $feedbackProjectRoot -WindowStyle Hidden
