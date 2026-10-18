## 这个 PR 做了什么

（一句话）

## 关联 Issue

Closes #

## 自检（必须全部通过）

```powershell
cmd /c build.bat
.\build\canticle.exe am-test
.\build\canticle.exe am-source
.\build\canticle.exe am-precedence
```

- [ ] 四项全部通过（贴输出）

## 如果你改了代码

- [ ] 已跑 `.\mutate.ps1`，全部 CAUGHT（贴结果）
- [ ] 如果新增了测试套件，已在 `mutate.h` 里加了对应变异

## 如果你改了性能相关的东西

| | 改前 | 改后 | 复现命令 |
|---|---|---|---|
| | | | |

**请给原始数字，不要只给百分比。**

## 指纹有没有变

```
.\build\canticle.exe fingerprint --quiet
```

（如果变了，说明改了状态量——在下面说明是哪个）
