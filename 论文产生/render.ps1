# ===========================================================================
#  render.ps1 -- 把 measured.json 的数字注入论文正文
#
#  论文源文件里的 {{key}} 会被替换成实测值。这样做的原因：
#  如果正文直接写字面量，它就成了需要手工保持同步的第二份事实，
#  而本工程已四次因为这种清单分叉而返工（判据 2）。
#
#  用法:  powershell -File 论文产生/render.ps1
#  输入:  论文产生/*.md.tpl   +  论文产生/measured.json
#  产物:  论文产生/*.md
# ===========================================================================

$ErrorActionPreference = 'Stop'
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$jsonPath = Join-Path $dir 'measured.json'

if (-not (Test-Path $jsonPath)) {
  Write-Error "缺少 measured.json，先运行 collect.ps1"
  exit 1
}
$data = Get-Content $jsonPath -Raw | ConvertFrom-Json

# 展平：sweep 是数组，单独渲染成表格，其余都是标量
$vars = [ordered]@{}
foreach ($p in $data.PSObject.Properties) {
  if ($p.Name -eq 'sweep') { continue }
  if ($p.Value -is [System.Management.Automation.PSCustomObject]) { continue }
  $vars[$p.Name] = $p.Value
}

# 生成 sweep 表格的 Markdown 文本，作为一个变量注入
$rows = @()
$rows += '| N | 三值 MB | int4 MB | fp16 MB | 三值 ms | int4 ms | fp16 ms | 胜者 |'
$rows += '|---|---|---|---|---|---|---|---|'
foreach ($s in ($data.sweep | Sort-Object N)) {
  $win = if ($s.winner -eq 't3') { '**三值**' } else { $s.winner }
  $rows += "| $($s.N) | $($s.t3_mb) | $($s.i4_mb) | $($s.f16_mb) | $($s.t3_ms) | $($s.i4_ms) | $($s.f16_ms) | $win |"
}
$vars['sweep_table'] = ($rows -join "`n")

# 嵌套键：{{int4_wins_at_ms.i4}} 这类带点的引用。
# 第一版只处理顶层标量，三个嵌套占位符被原样留在正文里 —— 渲染器没有报错，
# 因为它只检查 {{MISSING:...}}，而这些占位符根本没进替换分支。
# 一个「静默留下未替换内容」的模板引擎，和一个正确渲染的引擎，在成功路径上
# 输出相同；只有逐字检查产物才能发现差别。
if ($data.int4_wins_below_N) {
  $vars['int4_wins_at_ms.i4'] = $data.int4_wins_at_ms.i4
  $vars['int4_wins_at_ms.t3'] = $data.int4_wins_at_ms.t3
}

$tpls = Get-ChildItem -Path $dir -Filter '*.md.tpl'
if ($tpls.Count -eq 0) { Write-Host '没有找到 *.md.tpl'; exit 0 }

foreach ($t in $tpls) {
  $text = Get-Content $t.FullName -Raw
  $missing = @()
  $text = [regex]::Replace($text, '{{([a-zA-Z0-9_.]+)}}', {
    param($m)
    $k = $m.Groups[1].Value
    if ($vars.Contains($k)) { return [string]$vars[$k] }
    $script:missing += $k
    return "{{MISSING:$k}}"
  })
  # $missing 在 scriptblock 外不可见，改用二次扫描
  $left = [regex]::Matches($text, '{{MISSING:([a-zA-Z0-9_]+)}}');
  if ($left.Count -gt 0) {
    $names = ($left | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique) -join ', '
    Write-Error "$($t.Name): measured.json 里没有这些键: $names"
    exit 1
  }
  # 文件头的来源声明
  $stamp = "> 本文件的数字由 ``collect.ps1`` 于 $($data.generated) 实测注入。`n> 任何数字都可用 ``canticle <命令>`` 复现；正文不写字面量。`n"
  $outPath = Join-Path $dir ($t.BaseName)
  # 渲染后不许残留任何 {{...}}。第一版漏了这条，三个嵌套占位符就这样进了正文。
  if ($text -match '\{\{') {
    Write-Error "$($t.Name): 渲染后仍残留占位符，检查键名与 render.ps1 的嵌套处理"
    exit 1
  }
  ($stamp + "`n" + $text) | Set-Content -Path $outPath -Encoding UTF8
  Write-Host "  渲染 $($t.BaseName)"
}

Write-Host ''
Write-Host "  全部完成，共 $($tpls.Count) 篇"
