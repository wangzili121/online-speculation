# 冻结实验源码恢复

GSM8K和SPoC分别使用各自冻结的计算版本。本目录将原始 `.source.json` 中每个受保护 Python 文件，对应到仓库内实际源码文本，避免只保留哈希、依赖原服务器继续存在。

| 版本 | 受保护文件数 | 源码来源 |
| --- | ---: | --- |
| A / GSM v248 | 157 | `aux308/source233_bundle.json.gz` 中模型与benchmark；本目录补齐原生服务及Docker入口 |
| A / SPoC v292 | 157 | `relocate318/source292_A_bundle.json.gz` 中模型、benchmark与入口；本目录补齐原生服务 |
| B / SPoC v292 | 157 | 同一冻结模型，保留B机器实际使用的启动包装 |
| B / GSM v308 | 156 | 同一冻结GSM模型与benchmark，保留 `aux308/run_dense_v308.py` 的host入口 |

额外归档只包含原生服务的8个文件、A的GSM Docker入口2个、B的SPoC Docker入口2个，共12个不同文件；从正在使用的文件只读获取，逐个与已登记哈希核对。模型、benchmark及其余代码直接复用已有归档，不重复打包。原生服务来自冻结的OneServing集成版本，原文件头与内容完整保留。

```bash
# 只核对归档源码、哈希和语法，不运行模型
python3 recover_sources.py

# 恢复到一个尚不存在的新目录，并重新核对实际写出的每个文件
python3 recover_sources.py --output /tmp/orthrus-frozen-sources
```

每种版本分别恢复自己的工作目录、模型模块和原生服务文件，另附原始哈希清单。该入口**不安装镜像、CANN、PyPTO或模型权重，也不启动NPU任务**；它证明实验源码可以逐字节恢复，不能代替部署环境或性能验收。运行参数、数据及资源绑定仍以各实验的冻结计划和原始记录为准。
