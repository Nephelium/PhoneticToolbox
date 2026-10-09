param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [switch]$Apply,
    [ValidatePattern('^[a-zA-Z0-9-]+\.json$')][string]$ReportName
)
# Explicit reviewed targets only. Recheck file identities and live processes on every run.
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$plan = Get-Content -LiteralPath $Manifest -Encoding UTF8 | ConvertFrom-Json
if ($plan.schema -ne 'ptb-reviewed-cleanup/1' -or $plan.root -ne $root) { throw 'Cleanup manifest does not belong to this project.' }
$allowed = @('output\validation\', 'output\manual-work\', 'output\cleanup-20261008\', 'tools\manual-studio\test-output\')
if ($plan.manual_autosave_authorized -eq $true) {
    $allowed += @('manual\.studio\history\', 'manual\.studio\recovery\', 'manual\.studio\transactions\')
    $recoveryPath = [IO.Path]::GetFullPath((Join-Path $root $plan.keep_recovery.path))
    if (-not $recoveryPath.StartsWith((Join-Path $root 'manual\.studio\recovery') + '\', [StringComparison]::OrdinalIgnoreCase) -or
        (Get-FileHash -LiteralPath $recoveryPath -Algorithm SHA256).Hash -ne $plan.keep_recovery.sha256) { throw 'The retained latest recovery is missing or changed.' }
}
$protected = @('output\validation\p06\', 'output\validation\m01\', 'output\validation\m03-runtime\', 'output\validation\p03\')
function Assert-SafePath([string]$Path) {
    if (-not $Path.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Target is outside this project.' }
    $relative = $Path.Substring($root.Length + 1).TrimEnd('\') + '\'
    if (-not @($allowed | Where-Object { $relative.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) }).Count -or
        @($protected | Where-Object { $relative.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) }).Count) { throw 'Target is outside the approved scratch scope or is protected.' }
    $cursor = $Path
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Linked path retained: $cursor" }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
}
function Get-SafeFiles([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) { return ,$item }
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($Path)
    while ($pending.Count) {
        foreach ($child in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
            if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'A target contains a link.' }
            if ($child.PSIsContainer) { $pending.Push($child.FullName) } else { $child }
        }
    }
}
$results = [Collections.Generic.List[object]]::new()
foreach ($entry in $plan.targets) {
    $target = [IO.Path]::GetFullPath((Join-Path $root $entry.path)).TrimEnd('\')
    $result = [ordered]@{path=$target; bytes=$entry.bytes; reason=$entry.reason; status='retained'; error=$null}
    try {
        Assert-SafePath $target
        if ($target.StartsWith((Join-Path $root 'manual\.studio') + '\', [StringComparison]::OrdinalIgnoreCase)) {
            if ($target -eq $recoveryPath -or $recoveryPath.StartsWith($target + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'The latest recovery must be preserved.' }
            $lock = Get-Content -LiteralPath (Join-Path $root 'manual\.studio\lock.json') -Encoding UTF8 | ConvertFrom-Json
            if (-not $lock.released -and (Get-Process -Id $lock.pid -ErrorAction SilentlyContinue)) { throw 'The formal manual is open in an active author instance.' }
        }
        if (-not (Test-Path -LiteralPath $target)) { $result.status='already_absent' }
        else {
            $relative = $target.Substring($root.Length + 1).Replace('\', '/')
            $tracked = @(& git -C $root ls-files -- $relative)
            if ($LASTEXITCODE -ne 0 -or $tracked.Count) { throw 'Tracked or uninspectable content is retained.' }
            & git -C $root check-ignore -q -- $relative
            if ($LASTEXITCODE -ne 0) { throw 'Only ignored scratch is eligible.' }
            $files = @(Get-SafeFiles $target)
            $expected = @{}
            foreach ($file in $entry.files) { $expected[$file.path.Replace('/', '\')] = $file }
            if ($files.Count -ne $expected.Count) { throw 'The file inventory changed.' }
            foreach ($file in $files) {
                $key = $file.FullName.Substring($root.Length + 1)
                $match = $expected[$key]
                if (-not $match -or $file.Length -ne $match.bytes -or
                    (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash -ne $match.sha256) { throw "File changed since review: $key" }
            }
            $parent = [IO.Path]::GetDirectoryName($target)
            $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
            $active = @($processes | Where-Object {
                $_.ProcessId -ne $PID -and (
                    ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($target + '\', [StringComparison]::OrdinalIgnoreCase)) -or
                    ($_.CommandLine -and $_.CommandLine.IndexOf($parent, [StringComparison]::OrdinalIgnoreCase) -ge 0))
            })
            if ($active.Count) { throw 'A process still refers to the target or its test directory.' }
            if ($Apply) {
                # The absolute path and the whole file tree were verified above.
                Remove-Item -LiteralPath $target -Recurse -Force
                $result.status='deleted'
            } else { $result.status='ready' }
        }
    } catch { $result.error=$_.Exception.Message }
    $results.Add([PSCustomObject]$result)
}
$reportRoot = Join-Path $root 'output\maintenance'
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
$reportPath = Join-Path $reportRoot $(if ($ReportName) {$ReportName} elseif ($Apply) {'cleanup-results.json'} else {'cleanup-review.json'})
[ordered]@{schema='ptb-reviewed-cleanup-result/1'; apply=[bool]$Apply; targets=@($results.ToArray()); completedAt=(Get-Date).ToString('o')} |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding UTF8
$results | Group-Object status | Select-Object Name,Count | ConvertTo-Json -Compress
Write-Output $reportPath
if ($Apply -and @($results | Where-Object status -eq 'retained').Count) { exit 2 }
