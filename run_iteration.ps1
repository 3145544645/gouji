# ============================================================================
#  CANTICLE automatic iteration driver.
#
#  Runs the full measurement suite, appends a timestamped row to
#  experiments/results.csv, and prints a PASS/FAIL verdict against the
#  thresholds recorded in ITERATE.md.
#
#  Usage:  powershell -ExecutionPolicy Bypass -File run_iteration.ps1 -Tag label
# ============================================================================
param([string]$Tag = "unlabelled")

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Say($m) { Write-Host $m }

Say ("=== CANTICLE iteration: " + $Tag + " ===")
Say ("time: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

# ---- build ----
$b = cmd /c "build.bat" 2>&1
if ($LASTEXITCODE -ne 0) { Say "[ABORT] build failed"; $b | Select-Object -Last 20; exit 2 }
Say "[ok] build"

# ---- measurement 1: format crossover ----
Say ""
Say "--- sweep (format crossover, cold L2) ---"
$sweep = & ".\build\canticle.exe" sweep 2>&1
$sweepRows = $sweep | Select-String -SimpleMatch -Pattern '{"sweep"'
foreach ($r in $sweepRows) { Say $r.Line }

$last = $sweepRows | Select-Object -Last 1
$t3ms = ""; $i4ms = ""; $f16ms = ""; $spd4 = ""; $spd16 = ""
if ($last) {
  $j = $last.Line | ConvertFrom-Json
  $t3ms  = $j.sweep.t3_ms
  $i4ms  = $j.sweep.i4_ms
  $f16ms = $j.sweep.f16_ms
  $spd4  = [math]::Round($i4ms  / $t3ms, 3)
  $spd16 = [math]::Round($f16ms / $t3ms, 3)
  Say ("  largest-N speedups:  vs int4 = " + $spd4 + "x   vs fp16 = " + $spd16 + "x")
}

# ---- measurement 2: ternary training ----
Say ""
Say "--- ste (ternary training generalisation) ---"
$ste = & ".\build\canticle.exe" ste --steps 3000 --rq 8 2>&1
$ste | Select-String -Pattern "held-out|test  accuracy|train accuracy|PASS|FAIL" | ForEach-Object { Say ("  " + $_.Line) }
$stePass = ($ste | Select-String -SimpleMatch -Pattern "[PASS]").Count -gt 0

# ---- measurement 3: model forward ----
Say ""
Say "--- smoke (model forward / decode) ---"
$smoke = & ".\build\canticle.exe" smoke 2>&1
$smoke | Select-String -Pattern "generated|nan=|ternary planes" | ForEach-Object { Say ("  " + $_.Line) }
$toksPerSec = 0
$gm = $smoke | Select-String -Pattern "-> ([\d\.]+) tok/s"
if ($gm) { $toksPerSec = [double]$gm.Matches[0].Groups[1].Value }

# ---- record ----
$dir = Join-Path $root "experiments"
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$csv = Join-Path $dir "results.csv"
if (-not (Test-Path $csv)) {
  "timestamp,tag,t3_ms,i4_ms,f16_ms,speedup_vs_int4,speedup_vs_fp16,ste_pass,decode_tok_s" | Set-Content $csv
}
$row = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "," + $Tag + "," + $t3ms + "," + $i4ms + "," + $f16ms + "," + $spd4 + "," + $spd16 + "," + $stePass + "," + [math]::Round($toksPerSec,1)
Add-Content $csv $row
Say ""
Say ("[recorded] " + $row)

# ---- verdict ----
Say ""
Say "--- verdict (thresholds from ITERATE.md) ---"
$ok = $true
if ($spd4  -ne "" -and $spd4  -lt 1.5) { Say ("  FAIL  t3 vs int4 " + $spd4 + "x < 1.5");  $ok = $false } else { Say ("  pass  t3 vs int4 " + $spd4 + "x >= 1.5") }
if ($spd16 -ne "" -and $spd16 -lt 5.0) { Say ("  FAIL  t3 vs fp16 " + $spd16 + "x < 5.0"); $ok = $false } else { Say ("  pass  t3 vs fp16 " + $spd16 + "x >= 5.0") }
if (-not $stePass) { Say "  FAIL  ternary training did not generalise"; $ok = $false } else { Say "  pass  ternary training generalised" }
if ($toksPerSec -lt 1000) { Say ("  FAIL  decode " + $toksPerSec + " tok/s < 1000"); $ok = $false } else { Say ("  pass  decode " + $toksPerSec + " tok/s >= 1000") }
Say ""
if ($ok) { Say "[ALL GATES PASS] iteration is a keeper"; exit 0 }
else     { Say "[GATES FAILED] investigate before keeping"; exit 1 }
