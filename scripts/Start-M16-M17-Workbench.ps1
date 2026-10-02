# Shared source workbench with recording and IPA Plus. No install, build or DDL.
$ErrorActionPreference = 'Stop'
$newModulesRoot = Split-Path -Parent $PSScriptRoot
$newModulesPreviousSources = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (@('packages/phonetic_core/src','backend/src','desktop/src') | ForEach-Object { Join-Path $newModulesRoot $_ }) -join ';'
    & (Join-Path $PSScriptRoot 'Start-Research-Workbench.ps1')
} finally {
    $env:PYTHONPATH = $newModulesPreviousSources
}
