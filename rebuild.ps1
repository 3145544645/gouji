# 按时间顺序重建全部提交，为缺失 AI-Provider 的补上声明。
# 用 git commit-tree 保持树、父、作者、时间不变，只改提交信息。
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\cc\canticle'

$shas = @(git rev-list --reverse HEAD)
Write-Host ('共 ' + $shas.Count + ' 个提交')

$parent = $null
$n = 0
foreach ($sha in $shas) {
  $n++
  $tree = (git rev-parse ($sha + '^{tree}')).Trim()
  $msg  = git log -1 --format=%B $sha | Out-String
  $msg  = $msg.TrimEnd()

  if ($msg -match '(?mi)^AI-Provider:') {
    $newmsg = $msg
    $action = 'keep'
  } else {
    $newmsg = 'AI-Provider: Claude' + [char]10 + [char]10 + $msg
    $action = 'FILL'
  }

  $tmp = [System.IO.Path]::GetTempFileName()
  [System.IO.File]::WriteAllText($tmp, $newmsg)

  $env:GIT_AUTHOR_NAME     = (git log -1 --format=%an $sha).Trim()
  $env:GIT_AUTHOR_EMAIL    = (git log -1 --format=%ae $sha).Trim()
  $env:GIT_AUTHOR_DATE     = (git log -1 --format=%aI $sha).Trim()
  $env:GIT_COMMITTER_NAME  = (git log -1 --format=%cn $sha).Trim()
  $env:GIT_COMMITTER_EMAIL = (git log -1 --format=%ce $sha).Trim()
  $env:GIT_COMMITTER_DATE  = (git log -1 --format=%cI $sha).Trim()

  if ($parent) {
    $new = (git commit-tree $tree -p $parent -F $tmp).Trim()
  } else {
    $new = (git commit-tree $tree -F $tmp).Trim()
  }
  Remove-Item $tmp -Force

  $short = $sha.Substring(0,7)
  Write-Host ('  ' + $n.ToString().PadLeft(2) + ' ' + $action + ' ' + $short + ' -> ' + $new.Substring(0,7))
  $parent = $new
}

Write-Host ''
Write-Host ('新 HEAD: ' + $parent)
git update-ref refs/heads/main $parent
Write-Host 'main 已指向新历史'
