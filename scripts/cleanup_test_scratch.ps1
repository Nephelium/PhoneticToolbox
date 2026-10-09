param(
    [Parameter(Mandatory = $true)][string]$RunDirectory,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$run = [IO.Path]::GetFullPath($RunDirectory).TrimEnd('\')
$allowed = @('output\validation', 'output\manual-work', 'tools\manual-studio\test-output')
if (-not @($allowed | Where-Object { $run.StartsWith((Join-Path $projectRoot $_) + '\', [StringComparison]::OrdinalIgnoreCase) }).Count) {
    throw 'Test run must be below an owned test output directory.'
}
function Assert-NoLinkedAncestor([string]$Path) {
    $cursor = $Path
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Link retained: $cursor" }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
}
Assert-NoLinkedAncestor $run
$marker = Get-Content -LiteralPath (Join-Path $run 'run.json') -Encoding UTF8 | ConvertFrom-Json
if ($marker.schema -ne 'ptb-test-run/1' -or $marker.state -ne 'ready') { throw 'Run is not ready for scratch cleanup.' }
if (-not (Test-Path -LiteralPath (Join-Path $run 'scratch-manifest.json') -PathType Leaf)) { throw 'Scratch evidence is missing.' }
$target = Join-Path $run 'scratch'
$receipt = [ordered]@{status='retained'; path=$target; reason=$null; completedAt=(Get-Date).ToString('o')}
try {
    Assert-NoLinkedAncestor $target
    $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    $caller = $processes | Where-Object ProcessId -eq $PID
    $owner = $processes | Where-Object ProcessId -eq $marker.owner_pid
    if ($owner -and $caller.ParentProcessId -ne $marker.owner_pid) { throw 'Run owner is still active; cleanup must be called by its finalizer.' }
    $descendants = [Collections.Generic.HashSet[int]]::new()
    [void]$descendants.Add([int]$marker.owner_pid)
    do {
        $added = $false
        foreach ($process in $processes) {
            if ($process.ProcessId -ne $PID -and $descendants.Contains([int]$process.ParentProcessId)) {
                if ($descendants.Add([int]$process.ProcessId)) { $added = $true }
            }
        }
    } while ($added)
    if ($descendants.Count -gt 1) { throw 'Owned child processes still exist; stop them before cleanup.' }
    foreach ($process in $processes) {
        if ($process.ProcessId -in @($PID, $marker.owner_pid)) { continue }
        if (($process.ExecutablePath -and $process.ExecutablePath.StartsWith($run + '\', [StringComparison]::OrdinalIgnoreCase)) -or
            ($process.CommandLine -and $process.CommandLine.IndexOf($run, [StringComparison]::OrdinalIgnoreCase) -ge 0)) {
            throw 'A process still refers to this test run.'
        }
    }
    if (Test-Path -LiteralPath $target) {
        # Inspect one level at a time so traversal never follows a junction.
        $pending = [Collections.Generic.Stack[string]]::new()
        $pending.Push($target)
        while ($pending.Count) {
            foreach ($item in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
                if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Scratch contains a link: $($item.FullName)" }
                if ($item.PSIsContainer) { $pending.Push($item.FullName) }
            }
        }
        if (-not $PlanOnly) { Remove-Item -LiteralPath $target -Recurse -Force }
    }
    $receipt.status = if ($PlanOnly) { 'planned' } else { 'deleted' }
} catch { $receipt.reason = $_.Exception.Message }
$receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $run $(if ($PlanOnly) {'cleanup-plan.json'} else {'cleanup.json'})) -Encoding UTF8
$receipt | ConvertTo-Json -Compress
if ($receipt.status -eq 'retained') { exit 1 }
