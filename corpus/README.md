# 语料获取

本目录的 JSON 语料**不纳入版本库**（体积大，且是第三方数据集）。
程序在缺少语料时会给出明确错误，不会静默退化为空记忆。

## 需要两个文件

| 文件 | 用途 | 体积 | 来源 |
|---|---|---|---|
| `clinc150.json` | 英文意图分类，151 个意图 / 23700 对 | 2.4 MB | [CLINC150](https://github.com/clinc/oos-eval) |
| `alpaca_zh.json` | 中文指令，48818 条 | 34 MB | [Alpaca 中文数据集](https://github.com/ymcui/Chinese-LLaMA-Alpaca) 的 `alpaca_data_zh.json` |

## 格式

程序只要求两个字段，不要求原始格式：

```json
[{"text": "read the file C:/tmp/a.txt", "intent": "read_file"}]
```

转换只需把原始文件映射成上面的两个字段。本项目不使用 Python，
故不提供转换脚本；这一步是刻意的——语料是输入，不是代码的一部分。

## 校验

放好文件后跑：

```powershell
.\build\canticle.exe am-realclinc --per 50
```

若语料缺失，程序会报错退出，而不会静默用空记忆跑出一堆「拒答」然后
让你误以为模型坏了。
