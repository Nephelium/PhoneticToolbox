param([string]$Project)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
try {
    $manualStudioRoot = $PSScriptRoot
    $manualStudioNode = (Get-Command node -ErrorAction Stop).Source
    if (-not (Test-Path -LiteralPath (Join-Path $manualStudioRoot 'node_modules/@tiptap/vue-3'))) { throw 'Manual Studio dependencies are missing. Run npm ci --ignore-scripts in tools/manual-studio.' }
    if (-not (Test-Path -LiteralPath (Join-Path $manualStudioRoot 'dist/index.html'))) {
        & (Get-Command npm.cmd -ErrorAction Stop).Source --prefix $manualStudioRoot run build
        if ($LASTEXITCODE -ne 0) { throw 'Manual Studio build failed. The editor was not started.' }
    }
    $manualStudioRuntime = Join-Path $manualStudioRoot '.runtime/launches'
    $null = New-Item -ItemType Directory -Path $manualStudioRuntime -Force
    $manualStudioLaunch = Join-Path $manualStudioRuntime ([Guid]::NewGuid().ToString())
    $manualStudioReady = $manualStudioLaunch + '.json'
    $manualStudioArguments = @(('"' + (Join-Path $manualStudioRoot 'server/start.mjs') + '"'), '--ready-file', ('"' + $manualStudioReady + '"'))
    if ($Project) { $manualStudioArguments += @('--project', ('"' + $Project + '"')) }
    $manualStudioProcess = Start-Process -FilePath $manualStudioNode -ArgumentList $manualStudioArguments -WorkingDirectory $manualStudioRoot -WindowStyle Hidden -RedirectStandardOutput ($manualStudioLaunch + '.stdout.log') -RedirectStandardError ($manualStudioLaunch + '.stderr.log') -PassThru
    $manualStudioDeadline = [DateTime]::UtcNow.AddSeconds(30)
    while (-not (Test-Path -LiteralPath $manualStudioReady)) {
        if ($manualStudioProcess.HasExited) { throw ('Manual Studio exited before becoming ready. Details: ' + $manualStudioLaunch + '.stderr.log') }
        if ([DateTime]::UtcNow -ge $manualStudioDeadline) { throw ('Manual Studio startup timed out. Details: ' + $manualStudioLaunch + '.stderr.log') }
        Start-Sleep -Milliseconds 100
    }
    $manualStudioResult = Get-Content -LiteralPath $manualStudioReady -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $manualStudioResult.ok) { throw $manualStudioResult.error }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
