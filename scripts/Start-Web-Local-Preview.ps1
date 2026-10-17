param([ValidateRange(1024,65535)][int]$Port=5173)
$ErrorActionPreference='Stop'
$previewRoot=Split-Path -Parent $PSScriptRoot
$previewCache=Join-Path $previewRoot 'output/local-web-preview/vite-cache'
$previousCache=$env:PTB_VITE_CACHE_DIR
try {
    $env:PTB_VITE_CACHE_DIR=$previewCache
    Push-Location -LiteralPath (Join-Path $previewRoot 'frontend')
    try { & npm run dev -- --port $Port --strictPort }
    finally { Pop-Location }
    exit $LASTEXITCODE
} finally {
    $env:PTB_VITE_CACHE_DIR=$previousCache
}
