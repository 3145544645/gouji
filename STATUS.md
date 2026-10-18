# CANTICLE 最终状态

日期：2026-10-16  ·  纯 C++/CUDA · 无 Python · 无 PyTorch · 无第三方库

---

## 交付内容

两条**独立**的技术路线，均已实测通过：

### 路线 A：三值 GEMM 推理加速（低层）

| 指标 | 实测 | 命令 |
|---|---|---|
| 三值 vs int4（268M 权重，冷 L2） | **1.84x** | `canticle sweep` |
| 三值 vs fp16 | **7.08x** | `canticle sweep` |
| 三值有效带宽 | **217 GB/s = 峰值的 95%**（实测拷贝 228.5 GB/s） | `canticle sweep` |
| 三值 vs int4（16384 行） | **1.72x** | `canticle sweep` |

机理：三种格式都跑满带宽，**三值每权重只搬 2 bit**，所以最快。

### 路线 B：联想记忆对话 + 工具调用（高层）★ 主路线

| 指标 | 实测 | 命令 |
|---|---|---|
| 保留集工具名准确率 | **100.00%**（1350/1350） | `canticle am-eval` |
| 保留集整句完全匹配 | **99.41%**（1342/1350） | `canticle am-eval` |
| 容量扩展（500 → 12000 条） | **全部 100%** | `canticle am-eval --data N` |
| 写入速度 | **14.7 µs/条，一次遍历，零梯度** | `canticle am-build` |
| 检索延迟 | **0.286 ms → 3501 查询/秒** | `canticle am-bench` |
| 检索带宽 | **387 GB/s**（超过 DRAM 拷贝峰值，L2 命中） | `canticle am-bench` |
| 端到端对话延迟 | **0.4 ~ 1.3 ms** | `canticle am-chat` |
| 算术正确率 | **5/5**（逐条核对） | 见下 |

---

## 为什么放弃 Transformer 路线（重要）

Transformer 路径在本机**训不动**，且已定位到**多个真实根因**，不是调参能解决的：

1. **训练前向与推理前向是两个函数**
   `canticle align` 实测：logits 最大差 **24.3**，相对差 **556%**，top-5 重叠 **0/5**。
   训练在优化一个函数，推理在评估另一个。
2. 修复该问题后，全模型训练 loss 稳定降到 0.22，但模型**塌缩成高频 token**，
   连训练集内的样本都无法复述。0.81 MiB（约 4M 参数）对「自然语言 → 工具调用」太紧。
3. 期间修掉 9 个真实缺陷（见 `experiments/FINDINGS.md`），其中两个是并发内存错误，
   伪装成「权重健康但 loss NaN」。

**转向联想记忆后，一次写入、零梯度即达 100% 泛化。**

---

## 联想记忆的机制

### 写入（Hebb 规则，非反向传播）

每个 prompt 编码为单位范数向量，**直接追加进记忆矩阵**。
没有优化、没有 epoch、没有超参。第 N+1 条只碰第 N+1 行。

### 编码（`src/am.cpp`，纯 CPU，零参数）

字符三元组特征哈希 + IDF 加权 + 保留维度。
**关键**：让相似文本天然相似。

编码器迭代了三次，前两次的失败同样有价值：

| 版本 | 表示 | 结果 | 失败原因 |
|---|---|---|---|
| v1 | 模型嵌入均值 | 17.7%（=随机） | 嵌入是随机的，随机向量的均值几乎恒定 |
| v2 | 随机索引（VSA） | 17.7% | D=4096、2% 稀疏下任两向量**内积≈0 且方差极小**，实测余弦全是 0.000，所有查询等距 |
| **v3** | **字符 n-gram 特征哈希** | **100%** | — |

**教训**：不相关的东西**正交**的表示，**无法**做最近邻检索——没有相似度梯度可循。

### 检索

`am_dot` 内核：**一个 warp 负责一条记忆**，lane 分摊维度 → 完全合并访存。
键以 fp16 存储，字节数减半。这两步把延迟从 1.294 ms 压到 0.286 ms（**4.4x**）。

### 槽位迁移（Slot Transfer）

神经部分决定**选哪条**，载荷从查询里**直接拷贝**而非重新生成。
路径、表达式这类必须精确的载荷，拷贝是**构造性正确**的。

---

## 端到端实测

```
$ canticle am-chat --prompt "what is 123 * 456"
canticle> <|tool|>calc {"expr":"123 * 456"}<|/tool|>
  -> 123 * 456 = 56088                       [0.7 ms]

$ canticle am-chat --prompt "list the directory C:/tmp"
canticle> <|tool|>list_dir {"path":"C:/tmp"}<|/tool|>
  -> [真实列出目录内容]                       [1.3 ms]

$ canticle am-chat --prompt "who are you"
canticle> I am CANTICLE, a tiny local model running on this GPU.
```

算术逐条核对（全部正确）：

| 输入 | 期望 | 实得 |
|---|---|---|
| what is 7 + 8 | 15 | **15** ✅ |
| what is 123 * 456 | 56088 | **56088** ✅ |
| what is 999 - 111 | 888 | **888** ✅ |
| what is 100 / 4 | 25 | **25** ✅ |
| what is 12 * 7 | 84 | **84** ✅ |

---

## 单元测试

```
$ canticle am-test
  same template  : 0.354      <- 同模板相似度
  diff template  : 0.138      <- 跨模板相似度
  self-similarity: 1.000
[PASS]  16/16 passed
```

测试存在的理由：载荷提取**差一个字符**会让算术**静默算错**
（`what is 7 + 8` 曾答成 `77 + 8 = 85`）。肉眼没看出来，测试看出来了。

---

## 命令速查

| 命令 | 作用 |
|---|---|
| `canticle sweep` | 三值 vs int4/fp16 性能 A/B |
| `canticle ste --steps 3000` | 三值 STE 训练机制验证 |
| `canticle am-test` | 编码器与载荷提取单元测试 |
| `canticle am-build --data 12000` | 一次性建立记忆库 |
| `canticle am-eval --data 6000` | 泛化评测（留出 20%） |
| `canticle am-bench` | 检索延迟与带宽 |
| `canticle am-chat --prompt "..."` | 单轮对话（真实执行工具） |
| `canticle am-chat` | 交互式 REPL |

## 已知边界

- **文件写入受会话沙箱限制**：工作区内可写，`C:/tmp`、`%TEMP%` 返回 Win32 err 5。
  工具会明确报告「outside the writable root」，不再让接手者去查代码。
- `canticle train` 在 `--seq 160` 或 `--data 8000+` 时会触发 `illegal memory access`，
  `--seq 128 --data 3000` 稳定。已在 `TrainerImpl::init` 加 ctx 校验拦截。
