# ===========================================================================
#  Mutation testing driver (PowerShell).
#
#  WHY NOT BATCH
#  -------------
#  The first driver was mutate.bat, and it failed four different ways:
#
#    * cmd reads a batch file LINE BY LINE, so editing it while it runs executes a
#      mixture of old and new content (defect 133)
#    * :mut fell through into the verdict block without an explicit goto :eof
#    * for /f could not capture the probe answer reliably inside a called subroutine
#    * errorlevel propagation through nested if(...) blocks silently lost the verdict
#      lines, so the counters incremented while printing nothing
#
#  None of those is about mutation testing. They are all about cmd. PowerShell has a
#  real loop, real variables and real string comparison, so the same logic is a dozen
#  readable lines.
#
#  The lesson is the one this project keeps relearning: when a tool fights back, the
#  answer is usually to stop using it for that job, not to add another workaround.
# ===========================================================================

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe  = Join-Path $root 'build\canticle.exe'
$env:TEMP = Join-Path $root 'build\tmp'
$env:TMP  = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$mutations = @(
  @{ id='path-licence-off';               suite='am-precedence' },
  @{ id='command-vocab-off';              suite='am-realclinc'  },
  @{ id='file-ref-bare-word';             suite='am-realclinc'  },
  @{ id='spec-loses-lead';                suite='am-turns'      },
  @{ id='spec-wrong-bucket';              suite='am-source'     },
  @{ id='containment-ignores-expr';       suite='am-precedence' },
  @{ id='unsupported-becomes-read';       suite='am-unsupported'},
  @{ id='floor-off';                      suite='am-realclinc'  },
  @{ id='multiaction-off';                suite='am-multi-act'  },
  @{ id='multiaction-every-conjunction';  suite='am-multi-act'  },
  @{ id='copy-longest-separator-off';     suite='am-test'       },
  @{ id='copy-destination-optional';      suite='am-test'       },
  @{ id='rewrite-always-returns-input';   suite='am-turns'      },
  @{ id='rewrite-ignores-lead';           suite='am-turns'      },
  @{ id='multi-arg-after-guards';         suite='am-para'       },
  @{ id='field-replace-first-quote';      suite='am-para'       }
)

function Run-Quiet([string]$cmd, [string[]]$cmdArgs) {
  $out = & $cmd @cmdArgs 2>&1
  return @{ code = $LASTEXITCODE; text = ($out -join "`n") }
}

Write-Host '=== mutation testing ==='
Write-Host ''

# ---- baseline: every suite must pass before any mutation -------------------
# Without this a suite that is already broken would "catch" every mutation.
Write-Host '-- baseline --'
$suites = $mutations | ForEach-Object { $_.suite } | Select-Object -Unique
foreach ($s in $suites) {
  $r = Run-Quiet $exe @($s)
  if ($r.code -ne 0) {
    Write-Host "  FAIL: $s already fails before mutation. Fix it first."
    exit 1
  }
}
Write-Host "  all $($suites.Count) suites pass unmutated"
Write-Host ''

$caught = 0; $holes = 0; $invalid = 0; $skipped = 0

foreach ($m in $mutations) {
  Write-Host "-- $($m.id)"

  $probe = (& $exe am-mutate --probe $m.id 2>&1 | Out-String).Trim()
  $baseline = ''
  if ($probe -and $probe -ne 'NOPROBE') {
    $baseline = (& $exe am-chat --prompt $probe 2>&1 | Out-String)
  }

  $apply = Run-Quiet $exe @('am-mutate','--apply',$m.id)
  if ($apply.code -ne 0) {
    Write-Host '  SKIP: anchor moved. The harness is stale, not the test.'
    & $exe am-mutate --restore $m.id 2>&1 | Out-Null
    Write-Host ''
    $skipped++
    continue
  }

  $build = & cmd /c "build.bat" 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Host '  SKIP: the mutation does not compile.'
    & $exe am-mutate --restore $m.id 2>&1 | Out-Null
    Write-Host ''
    $skipped++
    continue
  }

  # Did the mutation change ANY behaviour? Compare the WHOLE answer, not one line:
  # field-replace-first-quote truncates the JSON while keeping the same prefix.
  $muted = ''
  if ($baseline -ne '') {
    $muted = (& $exe am-chat --prompt $probe 2>&1 | Out-String)
  }
  $changed = ($baseline -ne '') -and ($muted -ne $baseline)

  $suite = Run-Quiet $exe @($m.suite)
  if ($suite.code -ne 0) {
    Write-Host "  CAUGHT: $($m.suite) failed as required."
    $caught++
  } elseif (-not $changed) {
    Write-Host '  INVALID: the mutation changed NO behaviour; the mutation is aimed wrong.'
    $invalid++
  } else {
    Write-Host "  HOLE: $($m.suite) passed WITH THE MUTANT IN PLACE."
    Write-Host '        The behaviour did change and the suite still passed.'
    $holes++
  }

  & $exe am-mutate --restore $m.id 2>&1 | Out-Null
  Write-Host ''
}

& cmd /c "build.bat" 2>&1 | Out-Null

Write-Host '-- verdict --'
Write-Host "  caught $caught  holes $holes  invalid $invalid  skipped $skipped"
if ($caught -eq 0) {
  Write-Host ''
  Write-Host '  FAIL: no mutation was applied at all. A harness that ran nothing'
  Write-Host '  reports the same thing as one that found nothing -- defect 105.'
  exit 1
}
if ($holes -gt 0) {
  Write-Host ''
  Write-Host '  Each hole is a behaviour the named suite does not actually check.'
  exit 1
}
if ($invalid -gt 0) {
  Write-Host ''
  Write-Host '  Each INVALID mutation is aimed at code that cannot affect behaviour.'
  Write-Host '  Re-aim it rather than looking for a missing test.'
  exit 1
}
Write-Host ''
Write-Host '  ALL PASS  every applied mutation was caught'
exit 0
