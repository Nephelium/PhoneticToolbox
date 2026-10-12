$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$taskEvidence = Join-Path $taskRoot 'output/validation/feedback-manager'
$taskEmulator = Join-Path $taskEvidence 'emulator-owned'
$taskOwner = Get-Content -LiteralPath (Join-Path $taskEmulator 'owner.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskAdb = Join-Path $env:LOCALAPPDATA 'Android/Sdk/platform-tools/adb.exe'
$taskRemoved = [Collections.Generic.List[string]]::new()
function Remove-OwnedArtifact([string] $path, [string] $within) {
    $resolved = [IO.Path]::GetFullPath($path)
    $boundary = [IO.Path]::GetFullPath($within).TrimEnd('\','/') + '\'
    if (-not $resolved.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) { throw "Artifact outside owned boundary" }
    if (Test-Path -LiteralPath $resolved) {
        $item = Get-Item -LiteralPath $resolved -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refuse artifact junction' }
        if ($item.PSIsContainer -and (Get-ChildItem -LiteralPath $resolved -Force -Recurse | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })) { throw 'Refuse nested artifact junction' }
        Remove-Item -LiteralPath $resolved -Force -Recurse
        $taskRemoved.Add($resolved)
    }
}
if ($taskOwner.serial -ne 'emulator-5580' -or [IO.Path]::GetFullPath($taskOwner.path) -ne [IO.Path]::GetFullPath($taskEmulator)) { throw 'Unexpected emulator owner' }
if ($taskOwner.state -eq 'active') {
    $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskOwner.pid)"
    if (-not $taskProcess -or $taskProcess.ExecutablePath -ne (Join-Path $env:LOCALAPPDATA 'Android/Sdk/emulator/emulator.exe').Replace('/','\') -or $taskProcess.CommandLine -notmatch '5580') { throw 'Emulator process ownership cannot be confirmed' }
    $taskChildren = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($taskOwner.pid)" | Where-Object { $_.ExecutablePath -like '*Android\Sdk\emulator\qemu\*' })
    & $taskAdb -s $taskOwner.serial emu kill
    if ($LASTEXITCODE -ne 0) { throw 'Owned emulator did not stop' }
    Wait-Process -Id $taskOwner.pid -Timeout 20 -ErrorAction SilentlyContinue
    if (Get-Process -Id $taskOwner.pid -ErrorAction SilentlyContinue) { throw 'Owned emulator is still running' }
    foreach ($taskChild in $taskChildren) { if (Get-Process -Id $taskChild.ProcessId -ErrorAction SilentlyContinue) { throw 'Owned emulator child is still running' } }
    $taskOwner.state = 'stopped'
    [IO.File]::WriteAllText((Join-Path $taskEmulator 'owner.json'), ($taskOwner | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
}
Remove-OwnedArtifact (Join-Path $taskEmulator 'avd') $taskEmulator
Remove-OwnedArtifact (Join-Path $taskEmulator 'user') $taskEmulator
Remove-OwnedArtifact (Join-Path $taskEmulator 'feedback-qa.png') $taskEmulator
$taskQaRoot = 'D:\PTB-Feedback-Manager-QA'
foreach ($taskId in @('253851b1b3434bf68dc109ecaf24d494')) {
    $taskFolder = Join-Path $taskQaRoot $taskId
    if (Test-Path -LiteralPath $taskFolder) {
        $record = Get-Content -LiteralPath (Join-Path $taskFolder 'owner.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($record.state -ne 'stopped' -or $record.scope -ne 'Owned final manager EXE acceptance') { throw 'Frozen QA is still active' }
        if (Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($taskFolder + '\',[StringComparison]::OrdinalIgnoreCase) }) { throw 'Frozen QA process remains' }
        Remove-OwnedArtifact $taskFolder $taskQaRoot
    }
}
if (Test-Path -LiteralPath $taskQaRoot) {
    foreach ($taskFolder in Get-ChildItem -LiteralPath $taskQaRoot -Directory) {
        $recordPath = Join-Path $taskFolder.FullName 'owner.json'
        if (-not (Test-Path -LiteralPath $recordPath)) { continue }
        $record = Get-Content -LiteralPath $recordPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($record.report -ne (Join-Path $taskEvidence 'windows-final-26249ecb29884648979bf6fbc38ef594')) { continue }
        if ($record.state -ne 'stopped' -or $record.scope -ne 'Owned final manager EXE acceptance') { throw 'Frozen QA is still active' }
        if (Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($taskFolder.FullName + '\',[StringComparison]::OrdinalIgnoreCase) }) { throw 'Frozen QA process remains' }
        Remove-OwnedArtifact $taskFolder.FullName $taskQaRoot
    }
}
foreach ($taskName in @('android-final-notifications.tmp','android-final-services.tmp','android-reboot-services.tmp','android-reboot-log.tmp')) { Remove-OwnedArtifact (Join-Path $taskEvidence $taskName) $taskEvidence }
Remove-OwnedArtifact (Join-Path $taskRoot 'output/build-feedback-manager/pyinstaller') (Join-Path $taskRoot 'output/build-feedback-manager')
[IO.File]::WriteAllText((Join-Path $taskEvidence 'cleanup.json'), (@{success=$true; removed=@($taskRemoved.ToArray()); emulatorStopped=$true; originalAvdsUntouched=$true; officialPackagesAndSigningRetained=$true} | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
Write-Output 'Owned emulator, AVD, private fixture, frozen QA copies and raw dumps cleaned; final packages and signing retained.'
