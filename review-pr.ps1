# ===========================================================================
#  review-pr.ps1 -- pull a pull request and tell the author exactly what is
#  wrong.
#
#  WHY A SCRIPT AND NOT A CHECKLIST
#  -------------------------------
#  I wrote 代码检查规范.md as a five-dimension checklist, then ran it from memory.
#  That is the same mistake this project has fixed six times: a procedure that
#  lives in someone's head is not a procedure. Each dimension here is a concrete
#  command whose output is quoted back, so a rejection cites evidence rather than
#  impressions.
#
#  WHAT IT DOES, IN ORDER
#  ---------------------
#    1. fetch the PR branch into a scratch worktree (main stays untouched)
#    2. build
#    3. run the four gates
#    4. run mutation testing if code changed
#    5. check the diff size and the AI-Provider lines
#    6. compare the fingerprint before and after
#    7. print a verdict with per-item reasons
#
#  WHAT IT CANNOT DO
#  -----------------
#  Decide whether a design is good. It reports what can be measured; the five
#  dimensions in 代码检查规范.md still need a human (or an AI) to apply.
#
#  USAGE
#    .\review-pr.ps1 -Number 7
#    .\review-pr.ps1 -Number 7 -Keep     # keep the worktree for inspection
# ===========================================================================

param(
  [Parameter(Mandatory=$true)][int]$Number,
  [switch]$Keep,
  [switch]$SkipMutate
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$results = [System.Collections.Generic.List[object]]::new()
function Item($name, $status, $detail) {
  $script:results.Add([pscustomobject]@{ Name = $name; Status = $status; Detail = $detail })
  $icon = switch ($status) { 'PASS' { '[ ok ]' } 'FAIL' { '[FAIL]' } 'WARN' { '[warn]' } default { '[ ?? ]' } }
  Write-Host ("  $icon " + $name.PadRight(34) + $detail)
}

Write-Host "=== reviewing PR #$Number ===" -ForegroundColor Cyan
Write-Host ''

# ---- 0. PR metadata -------------------------------------------------------
$pr = gh pr view $Number --json title,author,baseRefName,headRefName,body,additions,deletions,changedFiles,files 2>&1 | ConvertFrom-Json
if (-not $pr) { Write-Host 'cannot read PR'; exit 2 }
Write-Host ("  title : " + $pr.title)
Write-Host ("  author: @" + $pr.author.login)
Write-Host ("  diff  : +" + $pr.additions + " -" + $pr.deletions + " across " + $pr.changedFiles + " files")
Write-Host ''

$before = @{}

# ---- 1. fetch into a scratch worktree -------------------------------------
$wt = Join-Path $env:TEMP ("gouji-pr-" + $Number)
if (Test-Path $wt) { git worktree remove --force $wt 2>&1 | Out-Null }
git fetch origin ("pull/" + $Number + "/head:pr-" + $Number) --force 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Item 'fetch PR branch' 'FAIL' 'git fetch 失败，分支可能已删除'; }
git worktree add -f $wt ("pr-" + $Number) 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Item 'create worktree' 'FAIL' $wt; exit 2 }
Item 'fetch PR into worktree' 'PASS' $wt

Push-Location $wt
try {

  # ---- 2. build -----------------------------------------------------------
  $b = & cmd /c 'build.bat' 2>&1
  if ($LASTEXITCODE -eq 0) { Item 'build' 'PASS' ($b | Select-Object -Last 1) }
  else {
    Item 'build' 'FAIL' 'see build\cmake.log / build\build.log below'
    Write-Host ''; Write-Host '  --- last 15 lines of build output ---' -ForegroundColor Yellow
    $b | Select-Object -Last 15 | ForEach-Object { Write-Host ('    ' + $_) }
    $log = 'build\build.log'
    if (Test-Path $log) {
      Write-Host '  --- first errors ---' -ForegroundColor Yellow
      Get-Content $log | Select-String -Pattern 'error' | Select-Object -First 8 | ForEach-Object { Write-Host ('    ' + $_.Line.Trim()) }
    }
  }
  Write-Host ''

  # ---- 3. the four gates --------------------------------------------------
  $gates = @(
    @{ n='am-test';        m='passed|FAIL' },
    @{ n='am-source';      m='ALL PASS|FAILED|FAIL ' },
    @{ n='am-precedence';  m='ALL PASS|FAILED|FAIL ' },
    @{ n='am-entries';     m='ALL PASS|FAILED|FAIL |SKIP' }
  )
  foreach ($g in $gates) {
    $o = & '.\build\canticle.exe' $g.n 2>&1 | Out-String
    $code = $LASTEXITCODE
    $line = ($o -split "`n" | Where-Object { $_ -match $g.m } | Select-Object -Last 1)
    if (-not $line) { $line = '(no verdict line)' }
    if ($code -eq 0) { Item $g.n 'PASS' $line.Trim() }
    else {
      Item $g.n 'FAIL' $line.Trim()
      Write-Host '    --- failing checks ---' -ForegroundColor Yellow
      ($o -split "`n" | Where-Object { $_ -match '^\s*FAIL' } | Select-Object -First 10) | ForEach-Object { Write-Host ('      ' + $_.Trim()) }
    }
  }
  Write-Host ''

  # ---- 4. mutation testing, only if code changed --------------------------
  $codeChanged = ($pr.files | Where-Object { $_.path -match '\.(cpp|cu|h)$' } | Measure-Object).Count -gt 0
  if (-not $codeChanged) {
    Item 'mutate' 'WARN' 'no C++/CUDA file changed, not run'
  } elseif ($SkipMutate) {
    Item 'mutate' 'WARN' 'skipped by -SkipMutate (run it before merging)'
  } else {
    $mo = & powershell -NoProfile -File (Join-Path $wt 'mutate.ps1') 2>&1 | Out-String
    $mline = ($mo -split "`n" | Where-Object { $_ -match 'caught \d+' } | Select-Object -Last 1)
    if (-not $mline) { $mline = '(no verdict)' }
    if ($mo -match 'ALL PASS' -or $mo -match 'holes 0\s+invalid 0') { Item 'mutate' 'PASS' $mline.Trim() }
    else {
      Item 'mutate' 'FAIL' $mline.Trim()
      ($mo -split "`n" | Where-Object { $_ -match 'HOLE|INVALID' } | Select-Object -First 8) | ForEach-Object { Write-Host ('      ' + $_.Trim()) }
    }
  }
  Write-Host ''

  # ---- 5. AI-Provider on every commit -------------------------------------
  $range = 'origin/main..HEAD'
  $shas = @(git rev-list $range)
  $missing = @()
  foreach ($s in $shas) {
    $b2 = git log -1 --format='%B' $s 2>&1 | Out-String
    if ($b2 -notmatch '(?m)^AI-Provider:') { $missing += $s.Substring(0,7) }
  }
  if ($shas.Count -eq 0) { Item 'AI-Provider' 'WARN' 'no commits in range (branch identical to main?)' }
  elseif ($missing.Count -gt 0) { Item 'AI-Provider' 'FAIL' ($missing -join ', ') + ' 缺少声明' }
  else { Item 'AI-Provider' 'PASS' "$($shas.Count) 个提交都有声明" }

  # ---- 7b. the TEST-EVIDENCE block in the PR body -------------------------
  #
  #  CI (evidence.yml) also checks this, but only on the server. When I review
  #  locally I must see the same thing CI saw, otherwise a PR that failed
  #  evidence.yml could still get a "looks fine" from me. The first version of
  #  this script omitted the check entirely -- caught by grepping the script for
  #  the string it was supposed to look for, which is the cheapest possible test.
  {
    $body = [string]$pr.body
    $block = [regex]::Match($body, "(?s)=== TEST-EVIDENCE ===(.*?)=== END ===")
    if (-not $block.Success) {
      Item "TEST-EVIDENCE" "FAIL" "PR 正文里没有证据区块，请用模板填写"
    } else {
      $inner = $block.Groups[1].Value
      $need = @("build","am-test","am-source","am-precedence","mutate","fingerprint")
      $bad = @()
      foreach ($f in $need) {
        # [ \t]* NOT \s*: \s matches newlines, so an empty value would swallow the
        # NEXT line and the field would look populated. Measured: "am-source: " with
        # nothing after it captured "am-precedence: ALL PASS" and passed.
        # Same class as CRITERIA.md 4 (matching on a substring cannot tell two
        # different things apart) -- here the substring was a line break.
        $pat = "(?m)^" + [regex]::Escape($f) + ":[ \t]*(.*)$"
        $m = [regex]::Match($inner, $pat)
        if (-not $m.Success) { $bad += ($f + " (缺字段)"); continue }
        $v = $m.Groups[1].Value.Trim()
        if ($v -eq "") { $bad += ($f + " (空)"); continue }
        $lv = $v.ToLower()
        if ($lv -match "^(ok|pass|passed|通过|done|yes|y|good|works|n/a|na|none|-)$") {
          $bad += ($f + " ('" + $v + "' 不是证据)");
        }
      }
      if ($bad.Count -gt 0) {
        Item "TEST-EVIDENCE" "FAIL" ($bad -join "; ")
      } else {
        Item "TEST-EVIDENCE" "PASS" "六个字段都有实际输出"
      }
    }
  }
  # ---- 6. diff size -------------------------------------------------------
  $lines = $pr.additions + $pr.deletions
  if ($lines -gt 500) { Item 'diff size' 'WARN' "$lines 行 > 500，按规范应拆分" }
  elseif ($pr.changedFiles -gt 5) { Item 'diff size' 'WARN' "$($pr.changedFiles) 个文件 > 5，按规范应拆分" }
  else { Item 'diff size' 'PASS' "$lines 行 / $($pr.changedFiles) 个文件" }

  # ---- 7. fingerprint -----------------------------------------------------
  $fpNew = (& '.\build\canticle.exe' fingerprint --quiet 2>&1 | Out-String).Trim()
  Push-Location $root
  $fpOld = (& '.\build\canticle.exe' fingerprint --quiet 2>&1 | Out-String).Trim()
  Pop-Location
  if ($fpNew -eq $fpOld) { Item 'fingerprint' 'PASS' "$fpNew (未变)" }
  else { Item 'fingerprint' 'WARN' "$fpOld -> $fpNew （状态量变了，需作者说明）" }

} finally {
  Pop-Location
}

# ---- verdict ---------------------------------------------------------------
Write-Host ''
Write-Host '=== verdict ===' -ForegroundColor Cyan
$fails = @($results | Where-Object { $_.Status -eq 'FAIL' })
$warns = @($results | Where-Object { $_.Status -eq 'WARN' })

if ($fails.Count -gt 0) {
  Write-Host ''
  Write-Host "  [拒绝] $($fails.Count) 项失败：" -ForegroundColor Red
  foreach ($f in $fails) { Write-Host ("    - " + $f.Name + ": " + $f.Detail) }
  Write-Host ''
  Write-Host '  回复模板：' -ForegroundColor Yellow
  Write-Host '  ---'
  Write-Host '  这个 PR 暂时不能合并，原因如下：';
  foreach ($f in $fails) { Write-Host ("    * " + $f.Name + " -- " + $f.Detail) }
  Write-Host ''
  Write-Host '  修好后再推一次即可，CI 会重跑。';
  Write-Host '  参考 CONTRIBUTING.md 与 PROJECT-RULES.md。'
  Write-Host '  ---'
  $exit = 1
} elseif ($warns.Count -gt 0) {
  Write-Host ''
  Write-Host "  [通过，但有 $($warns.Count) 项需确认]" -ForegroundColor Yellow
  foreach ($w in $warns) { Write-Host ("    - " + $w.Name + ": " + $w.Detail) }
  $exit = 0
} else {
  Write-Host ''
  Write-Host '  [通过] 全部门禁通过，可以合并' -ForegroundColor Green
  $exit = 0
}

Write-Host ''
Write-Host '  CI 只检查「填了证据」，不检查「证据是真的」。'
Write-Host '  上面这一遍才是真实性验证。'

if (-not $Keep) {
  git worktree remove --force $wt 2>&1 | Out-Null
} else {
  Write-Host "  工作树保留在: $wt"
}

exit $exit
