# CANTICLE 迭代循环规程

> 每轮实验遵守本规程。**没有测量数据的改动不算进展。**

## 0. 铁律

1. 任何"更快/更好"的声明必须附实测数字，否则作废。
2. 每个数字必须能被独立复现：跑的命令 + 期望输出。
3. 失败实验照样记录（`experiments/FINDINGS.md`），负结果也是知识。
4. 一次只改一个变量，否则归因失效。

## 1. 每轮固定五步

| 步 | 动作 | 命令 | 完成标准 |
|---|---|---|---|
| ① 基线 | 记录当前最优指标 | `canticle bench` / `canticle ste` | 写入 `experiments/results.csv` |
| ② 假设 | 写一句话假设 + 预期数字 | 手写进 FINDINGS | 含"若成立，X 应 ≥ Y" |
| ③ 改动 | 只改一个变量 | 编辑源码 | `git diff` 单一关注点 |
| ④ 测量 | 跑三次取中位数 | 见 §2 | 三次结果写入 CSV |
| ⑤ 裁决 | 与基线比较 | — | 显著好则 `git tag iter-N`；否则回退 |

## 2. 测量命令

```powershell
# 内核吞吐（P1 首创点）
cd C:\Users\cc\canticle
cmd /c build.bat
.\build\canticle.exe sweep          # 格式交叉点，冷 L2
.\build\canticle.exe bench --N 8192 --K 4096 --iters 500

# 三值训练机制（P2）
.\build\canticle.exe ste --steps 3000 --rq 8

# 模型前向与解码速度（P3）
.\build\canticle.exe smoke
```

## 3. 当前最优基线（2026-10-07）

| 指标 | 值 | 命令 | 门槛 |
|---|---|---|---|
| 三值 vs int4 冷 L2 加速 | **1.89x** @268M 权重 | `sweep` | ≥1.5x |
| 三值 vs fp16 冷 L2 加速 | **7.37x** @268M 权重 | `sweep` | ≥5x |
| 三值带宽峰值 | **223.7 GB/s** (DRAM 峰值的 98% / 实测拷贝 228.5) | `sweep` N=65536 | ≥200 |
| STE 留出泛化 | **100% 准确率, loss 0.009x chance** | `ste` | <0.70x |
| 训练吞吐 | 7299 steps/s | `ste` | ≥3000 |
| 模型前向解码 | 2133 tok/s (1.8MB 模型) | `smoke` | ≥1000 |

## 4. 待验证假设队列（按预期收益排序）

| # | 假设 | 预期收益 | 验证命令 | 风险 |
|---|---|---|---|---|
| H1 | 扩大语料 + 更大 d_model 可让对话模型学会工具协议 | 工具调用正确率 ≥80% | `canticle train` | 数据合成质量 |
| H2 | 三值内核改用 `__hmax2` 位选择可再提速 | +15% | `sweep` | 需 PTX 调优 |
| H3 | L2 钉死 22MiB 对 >32MB 模型有实测收益 | 1.2x @64MB | `sweep`+persist | 当前无差距 |
| H4 | GQA (n_kv < n_head) 降低 KV 缓存带宽 | +10% 长上下文 | `smoke` ctx=512 | 需训练配合 |

## 5. 已知负结果（勿重复踩坑）

| 日期 | 尝试 | 结果 | 结论 |
|---|---|---|---|
| 10-07 | `(sum ids) mod V` 作 STE 验证任务 | 训练 81% / 测试 0% | 线性头表达不了环绕，任务设计错误 |
| 10-07 | `bucket(sum ids)` 作验证任务 | 训练 78% / 测试 0.78% | 512 个离散 token 学不出一维数值语义 |
| 10-07 | requant_every=1（每步重量化） | 0.909x uniform | 量化噪声压过学习信号 |
| 10-07 | 三值 4-bit→2-bit 直接替换（无位平面优化） | 比 int4 慢 12% | 位提取指令开销 > 节省的带宽 |
| 10-07 | `L2Window` 钉死 ≤22MiB 权重 | 1.029x | 权重本来就全在 32MiB L2 里，实验无效 |

## 6. 环境硬约束（踩过的坑）

- **路径必须纯 ASCII**：nvcc 在 `mini研发` 这类中文路径下会截断路径，编译失败。实体目录 `C:\Users\cc\canticle`，桌面 junction 指向它。
- **CUDA 13 移除了 `cudaDeviceProp::memoryClockRate` / `clockRate`**，改用 `nvidia-smi`。
- **`f16_to_f32(uint16_t)` 会被 `__half` 隐式转换坑**：`__half` → `float` → 截断成 `uint16_t`，数据静默损坏。已在 common.h 加 `__half` 重载防护。
- **`build.bat` 里 `if not exist build mkdir build` 不可删**：build 目录被清后日志重定向失败，脚本会谎报 [OK]。
- **此机 L2 持久化上限 = 22 MiB**（非 32 MiB），`cudaDevAttrMaxPersistingL2CacheSize` 为准。
