# 构屎 GOUJI — 纯 C++/CUDA 的三值 mini LLM

> **模型指纹：GOUJI-8D4E-953C-E915**　（`canticle fingerprint` 可复算；改任一状态量指纹必变）

> 在一台 8 GB 显存的笔记本上做高速推理对话模型。**无 Python、无 PyTorch、无外部依赖。**

## 一句话结论

把权重压到 **2 bit/权重**（三值 {-1,0,+1} 位平面），在 RTX 4070 Laptop 上
比 int4 快 **1.92x**、比 fp16 快 **7.65x**（N=65536，冷 L2）；
并用 **straight-through 估计**在纯 CUDA 里把这种离散权重训了起来（留出集 100% 泛化）。

**但它是交叉的**：N=512 时 int4 反而更快（权重 0.5 MB 完整驻留 L2，
瓶颈从带宽变成解包指令）。三值的优势只在**带宽受限的规模区间**内成立。

> 数字由 `论文产生/collect.ps1` 实测注入，正文不写字面量。
> 上面这几个值可用 `canticle sweep` 复现。

## 三条首创点

| # | 名称 | 机理 | 实测 |
|---|---|---|---|
| 1 | **位平面三值 GEMV** | W 属于 {-1,0,+1} 存为 (P,M) 两个位平面，w = P - M，2 bit/权重；warp 分摊字实现完全合并访存 | 268M 权重下 **1.89x** vs int4、**7.37x** vs fp16、**98%** 带宽 |
| 2 | **L2 常驻权重** | cudaAccessPolicyWindow(Persisting) 把三值权重钉进 L2 | 实测 L2 持久化上限 **22 MiB**；已自动启用 |
| 3 | **三值 STE 训练** | fp32 主权重 + 前向重量化 + 直通梯度，全在 CUDA 内完成 | 留出集 **100%** 准确率，loss **0.009x** 随机基线 |

## 构建

```powershell
git clone <this-repo> canticle
cd canticle
cmd /c build.bat          # 需要 CUDA 13.1 + MSVC 2022 + CMake + Ninja
```

**依赖**：nvcc 13.1、MSVC 14.44、CMake 3.24+、Ninja。无第三方 C++ 库。

**注意**：项目路径**必须是纯 ASCII**。nvcc 在中文路径下会把路径截断导致编译失败。

## 克隆后：哪些能立刻跑，哪些需要先准备

构建完就有 `build\canticle.exe`。但**不是所有命令都能直接跑**——
需要数据和需要记忆的那部分要先生成。

| 类别 | 命令 | 克隆后可直接跑？ |
|---|---|---|
| 指纹 / 自检 | `fingerprint` `am-test` `am-source` `am-precedence` `am-entries` `criteria` | **是**\* |
| GPU 基准 | `dev` `bench` `sweep` `ste` | **是**（只要有 GPU） |
| 纯语法测试（不读记忆） | `am-multi` `am-multi-act` `am-unsupported` | **是** |
| 自带内置语料 | `am-eval` | **是**（\`--data\` 用内置句对，不读 ckpt） |
| 需要记忆 | `am-para` `am-sem` `am-e2e` `am-turns` `am-bench` `am-fixtest` `am-learntest` `am-induce` `am-collapse` | 否，先 `am-build` |
| 需要语料 | `am-realclinc` `am-realcase` `am-hunt` `am-inject` `am-real` | 否，见 `corpus/README.md` |
| 需要 GPU 训练 | `train` `ftrain` `chat` `am-chat --learn` | 否，先 `train` |

### 三步跑通完整自检

```powershell
# 1. 建记忆（用内置语料，不需要下载）
.\build\canticle.exe am-build --data 6000

# 2. 自检：单元测试 + 单一真源 + 入口一致性
.\build\canticle.exe am-test
.\build\canticle.exe am-source

# 3. 效果：留出集准确率 + 检索速度
.\build\canticle.exe am-eval --data 6000
.\build\canticle.exe am-bench
```

\* `am-entries` 会在无记忆时**跳过**「两种拒答可区分」那一项并退出 1——
因为空记忆下两种拒答都是空串，比较它们等于什么都没比较。

### 缺失数据时的行为

**这一步是刻意设计的**：所有依赖记忆或语料的套件在数据缺失时
**报错退出（exit 1）并说明缺什么**，而不是静默用空记忆跑出一堆「拒答」
让你误以为模型坏了。

> 这条规则本身是被缺陷逼出来的：曾有四个套件在完全无记忆的机器上
> 报告 `ALL PASS`，因为它们各自比较了 0 个样本而 "0 失败" 恒为真。
> 现在每个这样的套件都有**最小工作量断言**。

## 命令

| 命令 | 作用 |
|---|---|
| canticle dev | 打印 GPU 属性（SM 数、L2 大小、持久化上限） |
| canticle bench --N 8192 --K 4096 | GEMV 吞吐 A/B（int4 / fp16 / 三值） |
| canticle sweep | 格式交叉点扫描（冷 L2，0.5 MB → 268 MB） |
| canticle ste --steps 3000 --rq 8 | 三值 STE 训练机制验证（泛化门禁） |
| canticle smoke | 模型前向 + 解码吞吐冒烟 |
| canticle train --steps 3000 --data 12000 | 全模型训练 |
| canticle chat <ckpt> --prompt "..." | 单轮对话 |
| canticle chat <ckpt> | 交互式 REPL |

## 实测基线（RTX 4070 Laptop, 8 GB, sm_89）

| 指标 | 值 | 复现命令 |
|---|---|---|
| 实测拷贝带宽 | **228.5 GB/s** | canticle sweep |
| 三值有效带宽（268M 权重） | **223.7 GB/s = 98% 峰值** | canticle sweep |
| 三值 vs int4 | **1.89x** | canticle sweep |
| 三值 vs fp16 | **7.37x** | canticle sweep |
| STE 留出泛化 | **100%** | canticle ste |
| 训练吞吐 | 7300 steps/s（小模型） | canticle ste |
| 解码吞吐 | **2133 tok/s**（1.8 MB 模型） | canticle smoke |
| 模型尺寸 | 0.81 MiB 三值权重 / 3.71 bit/权重 | canticle chat 启动行 |

## 工具调用协议

模型用单行、无歧义的标记输出工具调用，而不是自由格式 JSON：

```
<|tool|>read_file {"path":"C:/work/alpha.txt"}<|/tool|>
```

内置工具：read_file / write_file / list_dir / run_shell / calc。
解析器刻意宽容（扫描标记而非严格解析），因为小模型有时会把标记裹在散文里。

## 目录

```
canticle/
  include/canticle/   头文件 common quant kernels model ops train ftrain ckpt data
  src/                实现
    kernels.cu          三值与 int4 GEMV 内核（推理）
    model.cpp           量化、设备驻留、前向
    model_ops.cu        嵌入 rmsnorm rope attention SwiGLU
    ftrain.cu           GEMM / CE / AdamW（训练）
    ftrain_kernels.cu   前向与反传内核
    ftrain_main.cpp     训练主循环与主权重管理
    tokenizer.cpp       字节级 BPE（纯 C++）
    chat.cpp / chat_main.cpp   对话与工具层
    data.cpp            合成指令数据
  experiments/        实验记录与结果 CSV
  ITERATE.md          迭代循环规程
  run_iteration.ps1   自动迭代驱动
```

## 文档

- ITERATE.md — 每轮迭代的固定五步、门禁阈值、待验证假设队列
- experiments/FINDINGS.md — 全部实验结论与 bug 根因（含负结果）

## 明确不做

不做分布式、不做多模态、不追 SOTA 精度。**只追单位带宽下的推理吞吐。**
