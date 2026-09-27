param([string]$EvidenceFolder='output/validation/m14/visual')
# Read-only Office rendering of M14-owned synthetic artifacts. New COM instances only.
$ErrorActionPreference='Stop'
$m14Dir=Join-Path (Split-Path -Parent $PSScriptRoot) $EvidenceFolder
$m14Word=New-Object -ComObject Word.Application
try {
 $m14Word.Visible=$false
 $m14Word.DisplayAlerts=0
 foreach($m14File in Get-ChildItem -LiteralPath $m14Dir -Filter *.docx) {
  $m14Doc=$m14Word.Documents.Open($m14File.FullName,$false,$true)
  try { $m14Doc.ExportAsFixedFormat((Join-Path $m14Dir ($m14File.BaseName+'.pdf')),17) } finally { $m14Doc.Close(0) }
 }
} finally { $m14Word.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($m14Word) }
$m14Excel=New-Object -ComObject Excel.Application
try {
 $m14Excel.Visible=$false
 $m14Excel.DisplayAlerts=$false
 $m14Excel.AutomationSecurity=3
 foreach($m14File in Get-ChildItem -LiteralPath $m14Dir -Filter 同音字表*.xlsx) {
  $m14Book=$m14Excel.Workbooks.Open($m14File.FullName,0,$true)
  try { $m14Book.ExportAsFixedFormat(0,(Join-Path $m14Dir ($m14File.BaseName+'.pdf'))) } finally { $m14Book.Close($false) }
 }
} finally { $m14Excel.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($m14Excel) }
