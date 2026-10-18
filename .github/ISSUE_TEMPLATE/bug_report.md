---
name: 缺陷报告
about: 报告一个可以复现的问题
title: "[缺陷] "
labels: bug
---

## 现象

（一句话说清楚发生了什么）

## 复现

```powershell
# 尽量给能直接粘贴运行的命令
.\build\canticle.exe am-chat --prompt "..."
```

## 期望 vs 实际

| | |
|---|---|
| 期望 | |
| 实际 | |

## 环境

```
指纹（先跑 .\build\canticle.exe fingerprint）：
GPU：
CUDA 版本：
```

## 是否属于「域外误执行」

（它做了不该做的事，还是拒绝了能做到的事？前者优先级最高）
