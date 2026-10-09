param(
    [Parameter(Mandatory = $true)][string]$KeepBuild,
    [string]$KeepStage,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$distRoot = Join-Path $projectRoot 'dist'
$outputRoot = Join-Path $projectRoot 'output'
$stageRoot = Join-Path $outputRoot 'release-staging'
function Assert-NoLinkedAncestor([string]$Path) {
    $cursor = [IO.Path]::GetFullPath($Path)
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Linked packaging path: $cursor" }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
}
Assert-NoLinkedAncestor $distRoot
Assert-NoLinkedAncestor $outputRoot
Assert-NoLinkedAncestor $stageRoot
$keepBuildPath = (Resolve-Path -LiteralPath $KeepBuild).Path.TrimEnd('\')
if ([IO.Path]::GetDirectoryName($keepBuildPath) -ne $distRoot) {
    throw 'KeepBuild must be a direct child of the project dist directory.'
}
$buildInfo = Get-Content -LiteralPath (Join-Path $keepBuildPath 'build-info.json') -Encoding UTF8 | ConvertFrom-Json
if ([IO.Path]::GetFileName($buildInfo.exe) -ne $buildInfo.exe) { throw 'Invalid EXE entry.' }
$currentExe = Get-Item -LiteralPath (Join-Path $keepBuildPath $buildInfo.exe)
if ($currentExe.Length -le 0 -or $currentExe.Length -ne $buildInfo.bytes) {
    throw 'The new EXE has not completed.'
}
$keepWork = Join-Path $outputRoot ('build-' + [IO.Path]::GetFileName($keepBuildPath))
$processes = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ExecutablePath, CommandLine)
$candidates = [Collections.Generic.List[object]]::new()
$incomplete = [Collections.Generic.List[object]]::new()

foreach ($folder in Get-ChildItem -LiteralPath $distRoot -Directory) {
    if ($folder.FullName -eq $keepBuildPath) { continue }
    $infoPath = Join-Path $folder.FullName 'build-info.json'
    if (-not (Test-Path -LiteralPath $infoPath -PathType Leaf)) { continue }
    try {
        $oldInfo = Get-Content -LiteralPath $infoPath -Encoding UTF8 | ConvertFrom-Json
        if ($oldInfo.kind -notin @('local-only-preview', 'distributable-preview') -or
            [IO.Path]::GetFileName($oldInfo.exe) -ne $oldInfo.exe -or
            -not (Test-Path -LiteralPath (Join-Path $folder.FullName $oldInfo.exe) -PathType Leaf)) { continue }
        $candidates.Add([PSCustomObject]@{Path=$folder.FullName; Parent=$distRoot; Kind='old-exe'})
    } catch { continue }
}
foreach ($folder in Get-ChildItem -LiteralPath $outputRoot -Directory) {
    if ($folder.FullName -eq $keepWork -or $folder.Name -notlike 'build-*') { continue }
    $lifecycle = Join-Path $folder.FullName 'artifact-lifecycle.json'
    if (Test-Path -LiteralPath $lifecycle -PathType Leaf) {
        $state = Get-Content -LiteralPath $lifecycle -Encoding UTF8 | ConvertFrom-Json
        if ($state.schema -eq 'ptb-build-work/1' -and $state.state -ne 'completed') {
            $incomplete.Add([PSCustomObject]@{Path=$folder.FullName; State=$state.state; OwnerPid=$state.owner_pid; Reason='Incomplete build; inspect log and process before removing.'})
            continue
        }
    }
    $specs = @(Get-ChildItem -LiteralPath $folder.FullName -Filter '*.spec' -File)
    if ($specs.Count -eq 0 -or -not (Test-Path -LiteralPath (Join-Path $folder.FullName 'pyinstaller') -PathType Container)) { continue }
    # Spec, PyInstaller work tree and driver log establish build ownership.
    $logPath = Join-Path $folder.FullName 'build.log'
    if (-not (Test-Path -LiteralPath $logPath -PathType Leaf)) { continue }
    $candidates.Add([PSCustomObject]@{Path=$folder.FullName; Parent=$outputRoot; Kind='old-build'})
}
$stages = @(Get-ChildItem -LiteralPath $stageRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'package-report.json') -PathType Leaf } |
    Sort-Object LastWriteTimeUtc -Descending)
$keepStagePath = $null
foreach ($folder in Get-ChildItem -LiteralPath $stageRoot -Directory -ErrorAction SilentlyContinue) {
    $lifecycle = Join-Path $folder.FullName 'artifact-lifecycle.json'
    if (Test-Path -LiteralPath $lifecycle -PathType Leaf) {
        $state = Get-Content -LiteralPath $lifecycle -Encoding UTF8 | ConvertFrom-Json
        if ($state.schema -eq 'ptb-build-work/1' -and $state.state -ne 'completed' -and
            (-not $KeepStage -or $folder.FullName -ne [IO.Path]::GetFullPath($KeepStage))) {
            $incomplete.Add([PSCustomObject]@{Path=$folder.FullName; State=$state.state; OwnerPid=$state.owner_pid; Reason='Incomplete staging; inspect installer log and process before removing.'})
        }
    }
}
if ($KeepStage) {
    $keepStagePath = (Resolve-Path -LiteralPath $KeepStage).Path.TrimEnd('\')
    if ([IO.Path]::GetDirectoryName($keepStagePath) -ne $stageRoot) { throw 'KeepStage must be a direct release-staging child.' }
} elseif ($stages.Count -gt 0) {
    # A portable-only build must leave the latest installer available.
    $keepStagePath = $stages[0].FullName
}
foreach ($folder in $stages) {
    if ($folder.FullName -eq $keepStagePath) { continue }
    if (@($incomplete | Where-Object Path -eq $folder.FullName).Count) { continue }
    $candidates.Add([PSCustomObject]@{Path=$folder.FullName; Parent=$stageRoot; Kind='old-installer'})
}
if (Test-Path -LiteralPath (Join-Path $keepWork 'host-archive/host-archive.json') -PathType Leaf) {
    foreach ($folder in Get-ChildItem -LiteralPath $keepWork -Directory) {
        if (($folder.Name -eq 'host-archive-cache-mismatch' -or $folder.Name -like 'host-archive-cache-rejected-*') -and
            (Test-Path -LiteralPath (Join-Path $folder.FullName 'host-archive.json') -PathType Leaf)) {
            $candidates.Add([PSCustomObject]@{Path=$folder.FullName; Parent=$keepWork; Kind='old-cache'})
        }
    }
}

$deleted = [Collections.Generic.List[string]]::new()
$planned = [Collections.Generic.List[string]]::new()
$retained = [Collections.Generic.List[object]]::new()
foreach ($candidate in $candidates) {
    try {
        Assert-NoLinkedAncestor $candidate.Path
        $item = Get-Item -LiteralPath $candidate.Path -Force
        $target = (Resolve-Path -LiteralPath $candidate.Path).Path.TrimEnd('\')
        if ([IO.Path]::GetDirectoryName($target) -ne $candidate.Parent -or
            -not $target.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'Target is outside its approved packaging parent or is a link.'
        }
        $buildName = if ($candidate.Kind -eq 'old-build') { $item.Name.Substring(6) } else { $item.Name }
        $inUse = @($processes | Where-Object {
            ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($target + '\', [StringComparison]::OrdinalIgnoreCase)) -or
            ($_.CommandLine -and (
             ($_.CommandLine.IndexOf($target, [StringComparison]::OrdinalIgnoreCase) -ge 0) -or
             ($_.CommandLine -match ('build_v3_local_preview.*--name[ =]+["'']?' +
                 [regex]::Escape($buildName) + '(?:["'']|\s|$)'))))
        })
        if ($inUse.Count -gt 0) { throw 'Directory is in use by a running process.' }
        $pending = [Collections.Generic.Stack[string]]::new()
        $pending.Push($target)
        while ($pending.Count) {
            foreach ($child in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
                if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Directory contains a link; retained for explicit review.' }
                if ($child.PSIsContainer) { $pending.Push($child.FullName) }
            }
        }
        if ($PlanOnly) { $planned.Add($target) }
        else { Remove-Item -LiteralPath $target -Recurse -Force; $deleted.Add($target) }
    } catch {
        $retained.Add([PSCustomObject]@{Path=$candidate.Path; Reason=$_.Exception.Message})
    }
}
$report = [ordered]@{
    schema='ptb-build-cleanup/1'; keptBuild=$keepBuildPath; keptWork=$keepWork; keptInstaller=$keepStagePath
    mode=$(if ($PlanOnly) {'plan'} else {'cleanup'}); planned=@($planned.ToArray())
    deleted=@($deleted.ToArray()); retained=@($retained.ToArray()); incomplete=@($incomplete.ToArray()); completedAt=(Get-Date).ToString('o')
}
$reportPath = Join-Path $keepBuildPath $(if ($PlanOnly) {'cleanup-plan.json'} else {'cleanup-report.json'})
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportPath -Encoding UTF8
[PSCustomObject]@{Deleted=$deleted.Count; Planned=$planned.Count; Retained=$retained.Count; Incomplete=$incomplete.Count; Report=$reportPath} | ConvertTo-Json -Compress
