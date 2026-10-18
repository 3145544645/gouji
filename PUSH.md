# 推送到 GitHub

仓库已提交完成，本地有 **5 个提交、98 个文件、0.5 MB**。
推送只差**认证**这一步，而这一步必须由你本人完成（凭据不能代填）。

---

## 方式 A：用 gh CLI（推荐，一条命令建仓并推送）

```powershell
cd C:\Users\cc\canticle
gh auth login          # 选 GitHub.com -> HTTPS -> 浏览器登录
gh repo create gouji --public --source=. --remote=origin --push
```

`gh repo create` 会自动创建远程仓库、加 origin、推送当前分支。

## 方式 B：先建仓库再推（浏览器操作）

1. 打开 https://github.com/new
2. 仓库名填 `gouji`，**不要**勾选 README / .gitignore / LICENSE（本地已有）
3. 创建后回到命令行：

```powershell
cd C:\Users\cc\canticle
git remote add origin https://github.com/<你的用户名>/gouji.git
git branch -M main
git push -u origin main
```

首次推送会弹出浏览器窗口要你授权，授权后凭据由 Windows 凭据管理器保存。

## 方式 C：用 SSH

```powershell
cd C:\Users\cc\canticle
git remote add origin git@github.com:<你的用户名>/gouji.git
git branch -M main
git push -u origin main
```

需要本机已配置 SSH key 并添加到 GitHub。

---

## 推送前已经确认的事

| 检查项 | 结果 |
|---|---|
| 全新克隆能否构建 | **能**（删掉 build/ 后直接 `build.bat`，不设任何环境变量） |
| 克隆后自检 | `am-test` 73/73、`am-source` ALL PASS |
| 克隆后指纹 | `GOUJI-8D4E-953C-E915`，与主仓库一致 |
| 仓库体积 | 0.5 MB（`build/` 3.5 GB 已排除） |
| 本机绝对路径 | 无 |
| 密钥 / token / 私钥 | 无 |
| 邮箱 | 无 |
| 他人版权内容 | 无（`papers/` 里的他人博客已排除） |

## 推送后建议做的第一件事

在**另一台机器**（或删掉本地目录重新克隆）跑一遍：

```powershell
git clone https://github.com/<你的用户名>/gouji.git
cd gouji
cmd /c build.bat
.\build\canticle.exe fingerprint
```

指纹应当仍是 `GOUJI-8D4E-953C-E915`。若不同，说明有状态量依赖了本机环境——
那本身就是一个值得记录的缺陷。

---

## 关于仓库名与描述

建议的仓库描述：

> 构屎 GOUJI — 纯 C++/CUDA 的三值 mini LLM。2 bit/权重，零 Python；
> 联想记忆工具路由，零训练参数；143 个缺陷与 29 个变异测试记录在案。

建议的 topics：`cpp` `cuda` `ternary-quantization` `associative-memory` 
`tool-calling` `llm-inference` `no-python`

---

## 未包含在仓库里的东西（以及为什么）

| 内容 | 体积 | 原因 |
|---|---|---|
| `build/` | 3.5 GB | 构建产物，克隆后可重新生成 |
| `corpus/*.json` | 36 MB | 第三方数据集，见 `corpus/README.md` |
| `build/ckpt/*.am` | 106 MB | 模型检查点，`am-build` 可重建 |
| `papers/blog*.txt` | 57 KB | **他人的博客文章**，无授权且非本项目产物 |
| `san.txt` | 17 KB | compute-sanitizer 的帮助输出，非本项目产物 |

前三项都能用仓库里的命令重新生成，这是它们不进去的正当理由。
后两项的排除理由不同——**它们根本不是本项目的东西**。
