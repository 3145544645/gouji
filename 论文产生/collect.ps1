# ===========================================================================
#  collect.ps1 -- 采集全部论文引用的实测数字
#
#  为什么要有这个脚本
#  ------------------
#  论文里的每个数字都必须能用一条命令复现。手工抄写会产生第二份需要
#  保持一致的事实，而本工程的判据 2 已经证明：两份手工清单必然分叉。
#
#  所以这里的做法是：跑真实命令 -> 从输出里抽数字 -> 写 measured.json。
#  论文正文只引用 measured.json 里的键，不写字面量。数字变了，重跑本脚本
#  并重新渲染即可；数字没变，论文与代码不可能不一致。
#
#  用法:  powershell -File 论文产生/collect.ps1
#  产物:  论文产生/measured.json
# ===========================================================================

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$exe  = Join-Path $root 'build\canticle.exe'
$out  = Join-Path $root '论文产生\measured.json'

if (-not (Test-Path $exe)) { Write-Error "先构建: cd $root; .\build.bat"; exit 1 }

$env:TEMP = Join-Path $root 'build\tmp'
$env:TMP  = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$m = [ordered]@{}
$m['generated'] = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
$m['commit']    = 'working tree (no VCS)'

# ---- 路线 A: 三值 GEMM ----
Write-Host '[1/5] sweep (ternary vs int4 vs fp16)'
$sweep = @()
foreach ($line in (& $exe sweep 2>&1)) {
  if ($line -match '^\{"sweep":(.*)\}$') {
    $sweep += ($matches[1] | ConvertFrom-Json)
  }
}
if ($sweep.Count -eq 0) { Write-Error 'sweep 未输出可解析的 JSON'; exit 1 }
$m['sweep'] = $sweep

# 最大规模处的加速比 -- 论文的核心数字
$big = $sweep | Sort-Object N | Select-Object -Last 1
$m['speedup_vs_int4'] = [math]::Round($big.i4_ms / $big.t3_ms, 2)
$m['speedup_vs_fp16'] = [math]::Round($big.f16_ms / $big.t3_ms, 2)
$m['biggest_N']       = $big.N
$m['ternary_mb']      = $big.t3_mb
$m['int4_mb']         = $big.i4_mb
$m['fp16_mb']         = $big.f16_mb
$m['ternary_ms']      = $big.t3_ms
$m['int4_ms']         = $big.i4_ms
$m['fp16_ms']         = $big.f16_ms

# 交叉点：int4 在哪个 N 以下更快
$crossover = $sweep | Where-Object { $_.winner -eq 'int4' } | Sort-Object N | Select-Object -Last 1
if ($crossover) {
  $m['int4_wins_below_N'] = $crossover.N
  $m['int4_wins_at_ms']   = @{ t3 = $crossover.t3_ms; i4 = $crossover.i4_ms }
} else {
  $m['int4_wins_below_N'] = $null
}

# ---- 三值 STE 训练 ----
Write-Host '[2/5] ste (straight-through ternary training)'
$steRaw = (& $exe ste 2>&1) -join "`n"
if ($steRaw -match 'final\s+train loss\s*:\s*([0-9.]+)') { $m['ste_train_loss'] = [double]$matches[1] }
if ($steRaw -match 'held-out test loss\s*:\s*([0-9.]+)') { $m['ste_test_loss']  = [double]$matches[1] }
if ($steRaw -match 'test\s+accuracy\s*:\s*([0-9.]+)%')     { $m['ste_test_acc']   = [double]$matches[1] }
if ($steRaw -match 'ternary weights\s*:\s*(\d+) of (\d+)')  { $m['ste_ternary']    = "$($matches[1])/$($matches[2])" }

# ---- 路线 B: 联想记忆 ----
Write-Host '[3/5] am-bench (retrieval latency)'
$bench = (& $exe am-bench 2>&1) -join "`n"
if ($bench -match 'per query\s*:\s*([0-9.]+) ms\s*->\s*(\d+) queries/s') {
  $m['query_ms']   = [double]$matches[1]
  $m['queries_per_s'] = [int]$matches[2]
}
if ($bench -match 'key matrix\s*:\s*([0-9.]+) MB') { $m['key_matrix_mb'] = [double]$matches[1] }
if ($bench -match 'episodes\s*:\s*(\d+)')          { $m['episodes']      = [int]$matches[1] }

Write-Host '[4/5] am-eval (held-out accuracy)'
$eval = (& $exe am-eval --data 6000 2>&1) -join "`n"
if ($eval -match 'tool-name accuracy\s*:\s*([0-9.]+)%\s*\((\d+)/(\d+)\)') {
  $m['tool_acc']   = [double]$matches[1]
  $m['tool_hits']  = $matches[2]
  $m['tool_total'] = $matches[3]
}
if ($eval -match 'exact reply match\s*:\s*([0-9.]+)%\s*\((\d+)/(\d+)\)') {
  $m['exact_acc'] = [double]$matches[1]
}

Write-Host '[5/5] am-build (write speed)'
$build = (& $exe am-build --data 6000 2>&1) -join "`n"
if ($build -match 'stored (\d+) patterns in ([0-9.]+) ms\s*\(([0-9.]+) us/pattern') {
  $m['write_patterns'] = [int]$matches[1]
  $m['write_total_ms'] = [double]$matches[2]
  $m['write_us_each']  = [double]$matches[3]
}

# ---- 鲁棒性：真实语料上的域外误执行率 ----
Write-Host '[+] am-realclinc (out-of-scope rate, English corpus)'
$rc = (& $exe am-realclinc --per 50 2>&1) -join "`n"
if ($rc -match 'out-of-scope acted on\s*:\s*([0-9.]+)%') { $m['oos_en'] = [double]$matches[1] }
if ($rc -match 'in-scope refused\s*:\s*([0-9.]+)%')       { $m['inrefused_en'] = [double]$matches[1] }
Write-Host '[+] am-realclinc --chinese (Chinese corpus)'
$rcz = (& $exe am-realclinc --chinese --per 400 2>&1) -join "`n"
if ($rcz -match 'out-of-scope acted on\s*:\s*([0-9.]+)%') { $m['oos_zh'] = [double]$matches[1] }

# ---- 工程度量：测试与变异 ----
Write-Host '[+] mutation table'
$mut = @(& $exe am-mutate --list 2>&1)
$m['mutations'] = $mut.Count
$m['suites']    = ($mut | ForEach-Object { ($_ -split "`t")[1] } | Select-Object -Unique).Count

$m | ConvertTo-Json -Depth 6 | Set-Content -Path $out -Encoding UTF8

Write-Host ''
Write-Host "  写入 $out"
Write-Host "  speedup vs int4 : $($m.speedup_vs_int4)x  (N=$($m.biggest_N))"
Write-Host "  speedup vs fp16 : $($m.speedup_vs_fp16)x"
Write-Host "  query           : $($m.query_ms) ms ($($m.queries_per_s)/s)"
Write-Host "  tool accuracy   : $($m.tool_acc)%"
Write-Host "  mutations       : $($m.mutations)"
