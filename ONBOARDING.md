# 上手：拉下来第一件事

> 你刚把这个仓库拉到本地。**按顺序做完下面五步，再动手改任何东西。**
>
> 整页大约 10 分钟。跳过它的代价通常是一小时的返工。

---

## 第 0 步 · 先读哪几个文件（以及哪些绝对不要整份读）

这是最容易搞错的一件事，所以放最前面。

### 必须**整份读完**的（三份，约 5000 tokens）

| 顺序 | 文件 | 大小 | 为什么必须先读 |
|---|---|---|---|
| 1 | `AI-ONLY.md` | 5 KB | **不读完不要提交**。规则、例外、以及哪一半无法验证 |
| 2 | `CONTRIBUTING.md` | 8 KB | 提交前要做什么、变异测试是什么、注释怎么写 |
| 3 | `CRITERIA.md` | 7 KB | 8 条判据，**写代码前读一遍能省掉大部分返工** |

### **绝对不要整份加载**的

| 文件 | 大小 | 为什么 |
|---|---|---|
| `architecture.html` | **156 KB（约 4 万 tokens）** | 整份读会撑爆上下文，而且 90% 与你当前任务无关 |
| `src/am_main.cpp` | 102 KB | 同上。按需读函数，不要整份读 |
| `src/am.cpp` | 81 KB | 同上 |

### 架构文档的正确用法：**按章节查**

`architecture.html` 是一个单文件 HTML，用浏览器打开。它的章节：

| 章节 | 什么时候看 |
|---|---|
| §0 总览 | 第一次，理解两条路线 |
| §1 主线数据流 | 要改路由逻辑时 |
| §2.x 各层设计 | 要改对应层时 |
| §3 三大首创点 | 要论证「为什么这么做」时 |
| §5 命令地图 | 找命令时 |
| §6 缺口清单 | 找能做什么时（配合 `ROADMAP.md`） |
| **§7 已修缺陷** | **改任何东西之前，先搜有没有相关先例** |
| §8 沙箱边界 | 遇到「命令莫名失败」时 |

**§7 是最该用的**：143 个缺陷逐条记录，每条带根因。
你正在踩的坑，很可能在里面。用 grep 搜关键词，不要通读。

```bash
grep -i "你想找的关键词" architecture.html
```

### 什么时候读源码

| 任务 | 读什么 |
|---|---|
| 加一个工具 | `include/canticle/model.h` 的 `ToolSpec` + `src/chat.cpp` 的 `tool_specs()` |
| 改意图判定 | `src/am.cpp` 的 `infer_intent`（用 grep 定位，别通读） |
| 改载荷提取 | `include/canticle/context.h`（约 300 行，可整读） |
| 改拒答逻辑 | `src/am_main.cpp` 的 `query_reranked`（grep 定位） |
| 改多轮 | `src/am.cpp` 的 `Session::rewrite` |
| 改工具实现 | `src/chat.cpp`（30 KB，可整读） |

---

## 第 1 步 · 构建

```powershell
cmd /c build.bat
```

**要求**：nvcc 13.1、MSVC 14.44、CMake 3.24+、Ninja。**无第三方 C++ 库。**

**路径必须纯 ASCII。** nvcc 在中文路径下会截断路径导致编译失败。
如果你在中文目录下，`mklink /J C:\gouji D:\你的\中文\路径` 建一个英文 junction。

成功标志：

```
[OK] build\canticle.exe
```

> `build.bat` 会自己把 `TEMP` 指向 `build\tmp`。
> 这是修过的缺陷：nvcc 把中间文件写进 %TEMP%，在 TEMP 不可写的账户上
> cmake 报「配置失败」，看起来像工具链坏了。**不需要你手工设 TEMP。**

---

## 第 2 步 · 确认你的构建是完整的

```powershell
.\build\canticle.exe fingerprint
```

应当输出：

```
  指纹  GOUJI-8D4E-953C-E915
```

**如果指纹不同，说明你的构建和文档描述的不是同一个模型。**
可能原因：

| 现象 | 原因 |
|---|---|
| 指纹不同但源码没改 | 你在改过的分支上 |
| 指纹显示「不完整」 | 不在项目根目录运行 |
| 指纹相同 | 正常，继续 |

指纹由**状态量**算出（维度、意图数、工具数、阈值、判据数、变异数）。
它是「这是不是同一个模型」的答案，不是「能不能跑」的答案。

---

## 第 3 步 · 跑自检（三个，都必须过）

```powershell
.\build\canticle.exe am-test        # 73 项单元测试
.\build\canticle.exe am-source      # 单一真源 + 判据 + 论文 + AI 政策
.\build\canticle.exe am-precedence  # 规则优先级与上下文许可
```

**三个都显示 ALL PASS / passed 才算环境正常。**

> `am-source` 检查的东西比你想的多：
> 工具表与提示词是否一致、判据文档是否完整、论文数字是否过期、
> **每个测试套件是否有对应变异**、AI 政策是否在位。
> 它失败时**会告诉你怎么修**，读它的输出。

---

## 第 4 步 · 建一个记忆（想跑效果类命令才需要）

```powershell
.\build\canticle.exe am-build --data 6000
.\build\canticle.exe am-eval  --data 6000     # 应显示 100%
.\build\canticle.exe am-bench                 # 应显示 0.2~0.3 ms
```

**约 53 MB，几十秒。** 这一步生成本地记忆文件，不入库。

### 不建会怎样

依赖记忆的套件会**明确报错退出**，而不是静默通过：

```
  FAIL: no request produced an answer. The memory is missing or
  unreadable, so this run measured nothing.
  Build one first: canticle am-build --data 6000
```

> 这条规则是被缺陷逼出来的：曾有四个套件在完全无记忆的机器上
> 报告 `ALL PASS`，因为它们各自比较了 0 个样本而「0 失败」恒为真。

---

## 第 5 步 · 启用提交钩子（**不启用就别提交**）

```powershell
git config core.hooksPath .githooks
```

**这一步是手动的**，因为 git 出于安全不自动执行仓库里的钩子。

启用后，提交信息里必须有：

```
AI-Provider: Claude
```

否则：

```
  [RECHAZADO] 提交信息缺少 AI 来源声明。
```

---

## 检查清单：你可以开始了吗

全部打勾再动手：

- [ ] 读完了 `AI-ONLY.md`、
      `CONTRIBUTING.md`、
      `CRITERIA.md` 三份
- [ ] `build.bat` 输出 `[OK]`
- [ ] `fingerprint` 显示 `GOUJI-8D4E-953C-E915`（或你已知道为什么不同）
- [ ] `am-test` / `am-source` / `am-precedence` 三个都过
- [ ] （若跑效果类）`am-build` 完成
- [ ] `git config core.hooksPath .githooks` 已执行

---

## 最常见的四个坑

| 坑 | 症状 | 解法 |
|---|---|---|
| **整份读 architecture.html** | 上下文爆掉，回答开始丢细节 | 只 grep 你要的章节 |
| **在中文路径下构建** | nvcc 报奇怪的路径错误 | 建英文 junction |
| **没建记忆就跑效果套件** | 一堆 FAIL 说「measured nothing」 | `am-build --data 6000` |
| **忘了启用钩子** | 提交被拒，或（更糟）以为规则没生效 | `git config core.hooksPath .githooks` |

---

## 下一步

- 想知道能做什么 → `ROADMAP.md`
- 想立刻动手 → `CONTRIBUTING.md` 的「最容易上手的切入点」
- 想理解某个设计为什么这样 → `architecture.html` 的 §7 搜关键词
- 想提反例 → `.github/ISSUE_TEMPLATE/counterexample.md`（**这是最受欢迎的贡献**）
