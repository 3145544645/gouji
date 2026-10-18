# CANTICLE — 使用手册

纯 C++/CUDA，无 Python、无 PyTorch、无第三方库。
GPU：RTX 4070 Laptop（8 GB，sm_89）。

---

## 1. 五分钟上手

```powershell
cd <你的路径>\canticle
cmd /c build.bat                    # 编译（需 CUDA 13.1 + MSVC 2022 + CMake + Ninja）

build\canticle.exe am-build --data 12000    # 建立记忆库（约 0.2 秒）
build\canticle.exe am-chat --prompt "what is 123 * 456"
```

输出：

```
canticle> <|tool|>calc {"expr":"123*456"}<|/tool|>
  -> 123*456 = 56088
```

对话模式（不带 `--prompt` 即进入 REPL）：

```powershell
build\canticle.exe am-chat
```

---

## 2. 全部命令

运行 `build\canticle.exe` 不带参数即打印帮助。

### 联想记忆（对话 + 工具，主路线）

| 命令 | 说明 |
|---|---|
| `am-build [--data N] [--am FILE]` | 一次性写入记忆库。N = 合成样本数（默认 6000） |
| `am-eval [--data N] [--diag]` | 留出 20% 评测泛化。`--diag` 打印失败案例 |
| `am-bench` | 检索延迟与带宽 |
| `am-ask --prompt "TEXT"` | 只检索，不执行工具 |
| `am-chat [--prompt "TEXT"] [--debug]` | **完整循环**：检索 → 执行工具 → 返回结果 |
| `am-test` | 编码器与载荷提取的单元测试 |

### 三值内核与训练

| 命令 | 说明 |
|---|---|
| `sweep` | 冷 L2 下扫描 0.5 MB → 268 MB 的格式交叉点 |
| `bench [--N n --K k --iters i]` | 三值 vs int4 vs fp16 的 GEMV A/B |
| `ste [--steps n --rq n]` | 三值 STE 训练机制验证（泛化门禁） |
| `train [--steps n --seq n --data n]` | 全模型三值训练 |
| `smoke` | 模型前向 + 解码吞吐 |

### 诊断

| 命令 | 说明 |
|---|---|
| `dev` | GPU 属性与 L2 持久化预算 |
| `align` | 训练前向 vs 推理前向的一致性 |
| `tokdiag` | 分词器 save/load 往返 |
| `dump` | 打印打包后的训练序列 |
| `chat [--prompt] <ckpt>` | Transformer 对话（已知受限，见 STATUS.md） |

---

## 3. 记忆库文件

`build\ckpt\canticle.am` — 13500 条记忆，约 106 MB。

格式（自描述，纯二进制）：

```
magic "CAM2" (4B) | D:int32 | N:int32
N x D fp32 键矩阵
每条: len:u32 后跟 prompt 字节, len:u32 后跟 reply 字节
```

**加载时不需要 tokenizer 或模型检查点** —— 编码器是纯字符哈希，零参数。

---

## 4. 工具协议

模型用单行标记输出调用：

```
<|tool|>read_file {"path":"C:/work/alpha.txt"}<|/tool|>
```

| 工具 | 参数 | 说明 |
|---|---|---|
| `read_file` | `path` | 读文本，超 4000 字节截断 |
| `write_file` | `path`, `content` | 自动建父目录 |
| `list_dir` | `path` | 列出目录，最多 60 项 |
| `run_shell` | `cmd` | 经 `cmd /c` 执行，捕获输出 |
| `calc` | `expr` | 递归下降求值 `+ - * / ( )` |

解析器刻意宽容（扫描标记而非严格解析），因为小模型有时会把标记裹在散文里。

---

## 5. 已知边界

| 边界 | 说明 |
|---|---|
| **文件写入受沙箱限制** | 工作区内可写；`C:/tmp`、`%TEMP%` 返回 Win32 err 5。工具会明确报告「outside the writable root」 |
| **`train --seq 160` 崩溃** | `illegal memory access`。`--seq 128 --data 3000` 稳定，已在 `init` 加校验拦截 |
| **Transformer 路线不可用** | 见 STATUS.md：训练与推理是两个函数（差异 556%），修复后仍塌缩 |
| **编码器无语义理解** | 纯字符 n-gram，对同义改述（「读取文件」vs「read the file」）无能为力 |
| **路径必须是纯 ASCII** | nvcc 在中文路径下会截断路径导致编译失败 |

---

## 6. 常见问题

**Q: `am-build` 需要多久？**
A: 13500 条约 0.2 秒（14.5 µs/条，单次遍历，零梯度）。

**Q: 加更多记忆会变慢吗？**
A: 线性变慢，但余量很大。13500 条时 0.29 ms（3465 查询/秒）。按此推算 10 万条约 2 ms。

**Q: 为什么不用训练？**
A: 因为不需要。Hebb 外积一次写入即达 100% 留出泛化，而梯度下降路线在本机训不动。

**Q: 三值内核和联想记忆怎么结合？**
A: 目前联想记忆的键是 fp16。改三值可再减半带宽（理论再快约 2×），是待验证的下一步。

---

## 7. 目录

```
canticle/
  include/canticle/        头文件
    am.h                     联想记忆接口
    common.h                 基础类型与容错宏
    kernels.h                推理内核声明
    model.h                  模型、分词器、工具协议
    ops.h                    算子声明
    train.h / ftrain.h       训练
    ckpt.h / data.h          检查点与数据
  src/
    am.cpp                   编码器（纯 CPU，零参数）
    am_kernels.cu            检索内核
    am_main.cpp              记忆主程序与对话循环
    amtest.cpp               单元测试
    kernels.cu               三值 / int4 GEMV
    sweeper / model / ops    Transformer 路线
    chat.cpp                 工具实现
  experiments/FINDINGS.md   全部实验结论与 bug 根因
  STATUS.md                 最终状态与实测数字
  ITERATE.md                迭代规程与门禁
```
 
