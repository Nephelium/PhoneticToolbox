$ErrorActionPreference = 'Stop'
$m17AuthorRoot = Split-Path -Parent $PSScriptRoot
$m17AuthorNode = (Get-Command node -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath (Join-Path $m17AuthorRoot 'frontend/node_modules/vite'))) { throw '项目开发环境缺失，请恢复既有前端环境。' }
Start-Process -FilePath $m17AuthorNode -ArgumentList @('"' + (Join-Path $PSScriptRoot 'start_m17_author.mjs') + '"') -WorkingDirectory $m17AuthorRoot -WindowStyle Hidden
