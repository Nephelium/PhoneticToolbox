# Module shortcut; all launch behavior lives in Start-Research-Workbench.ps1.
param([string]$ComponentRoot = '', [switch]$CheckOnly, [switch]$PrepareOnly)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'Start-Research-Workbench.ps1') -Module 'M03' @PSBoundParameters
