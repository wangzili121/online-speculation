# DSV4.1 W4A8 内网双机部署

在两台各有 8 张 Ascend A2 的机器上，用 vLLM-Ascend 启动同一个 Orthrus 服务。每台机器各跑一个 DP rank，节点内用 TP8，MoE 专家通过 EP16 跨节点通信。只有节点 0 提供 API。Engram 表放 CPU。两台机器上的 `npu-smi` 都只显示各自的 8 张卡，集群是否连通以 16-rank HCCL 实测为准。 [官方 A2 双机配置](https://docs.vllm.ai/projects/ascend/zh-cn/v0.32.0rc/tutorials/models/DeepSeek-V4.1-Flash.html#55-多节点共置部署)

## 准备

- 两台机器都使用同一套 W4A8 权重，默认目录为 `/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8`。共享盘可直接使用。若各用本地盘，则两边均需完整权重；用 `MODEL_HOST` 可覆盖默认目录。
- 两台机器都预装配套的 ARM64 vLLM-Ascend 镜像。脚本使用 `quay.io/ascend/vllm-ascend@sha256:f040ca190fe5c5864a5491b2cbe19e687966ae8609b8a68fccc2c1ad1d260cbe`，对应 `nightly-releases-v0.32.0rc`。如果镜像通过 `docker load` 导入，且只保留镜像 ID，设置 `PINNED_LOCAL_IMAGE_ID=sha256:5a70e7cdbe81b2b2addad33cf31db6f8e8163fa4b510ff307a8864d9db36b3a2`。两边要能获准使用 Docker 和本节点的 8 张卡。
- 节点之间需要可用的 HCCL 数据面。先找出两个节点互相访问的内网 IP。启动脚本默认按对端 IP 的路由推断网卡，可用 `NIC0`、`NIC1` 指定。SSH 只负责传源码和启动容器，使用 HCCL 通信的 IP 可以与 SSH 地址不同。默认 rendezvous 端口是探针 `13401`、vLLM DP `13400`，API 是 `8901`。其他通信端口按公司集群的 HCCL 和防火墙配置开放。

## 从节点 0 启动两台

能访问本代码仓时，在节点 0 获取代码。

```bash
git clone --single-branch -b feature/orthrus-dsv41-pypto https://github.com/wangzili121/online-speculation.git
cd online-speculation/online-speculation
```

离线时，将交接的 tar.gz 上传到节点 0 并解压，进入解压目录的 `online-speculation/` 子目录。启动命令会通过 SSH 把所需源码复制到节点 1。容器使用本地镜像、权重和源码，启动时可在无外网的环境运行。

```bash
export NODE0_IP=10.0.0.10
export NODE1_IP=10.0.0.11
export NODE1_SSH=user@10.0.0.11
export ACK_EXCLUSIVE_NODE=1

# 可选。测 16 个 rank 的 AllReduce 和 AllToAll，不加载模型。
bash scripts/dsv41_two_node.sh probe

# 启动两节点。首次可从目标 Attention 导出 BF16 初始化 head。
bash scripts/dsv41_two_node.sh up
bash scripts/dsv41_two_node.sh status
```

默认初始化 head 写入 `/workspace/models/orthrus-dsv41-head-v1`，需要约 10 GiB 磁盘。若模型目录只读，提前把 `HEAD_HOST` 设成两边都可访问的可写绝对路径。已经有训练好的 head 时，设置 `HEAD_HOST=/path/to/head INIT_HEAD=0 REQUIRE_TRAINED_HEAD=1`；启动前会检查它与目标权重及 tokenizer 的哈希是否匹配。初始化 head 用于验证链路，接受率和加速需要训练后的 head。

`status` 显示两边的容器状态和节点 0 的 API 健康状态。只有模型加载、CPU Engram 初始化和跨机通信都完成后，`/health` 才会正常返回。用 `docker logs -f dsv41-orthrus-node0`、`docker logs -f dsv41-orthrus-node1` 分别在对应节点查看日志。用 `bash scripts/dsv41_two_node.sh down` 停掉本次服务。要先跑基模排查时，在 `up`、`status`、`down` 前加 `MODE=target`。

```bash
curl -fsS "http://$NODE0_IP:8901/v1/models"
curl -fsS "http://$NODE0_IP:8901/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-orthrus","prompt":"Continue the list of positive integers starting at 1.","temperature":0,"max_tokens":64}'
```

## 无法从节点 0 SSH 到节点 1 时

将代码包复制到两台机器。先在节点 0、再在节点 1 各执行下列命令，按本机调整 `NODE_RANK`、`LOCAL_IP` 和网卡。要手工跑 HCCL 探针，把 `MODE=orthrus` 改为 `MODE=probe`，两个容器均退出且日志有 16 个 rank 的通过结果即表示通信成功。

```bash
NODE_RANK=0 NODE0_IP=10.0.0.10 LOCAL_IP=10.0.0.10 NIC_NAME=eth1 \
  ACK_EXCLUSIVE_NODE=1 MODE=orthrus INIT_HEAD=1 REQUIRE_TRAINED_HEAD=0 \
  bash scripts/deploy_dsv41_node.sh
```

如果 `/workspace/models` 仅在公司预先启动的容器里可见，可以在各自的容器内直接运行服务脚本。容器需要配套镜像、8 个可见设备，以及可直连对端的网络。同样先在节点 0 启动，再在节点 1 用自己的 `NODE_RANK`、`LOCAL_IP` 和 `NIC_NAME` 启动。

```bash
NODE_RANK=0 NODE0_IP=10.0.0.10 LOCAL_IP=10.0.0.10 NIC_NAME=eth1 \
  MODEL=/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8 \
  HEAD=/workspace/models/orthrus-dsv41-head-v1 INIT_HEAD=1 REQUIRE_TRAINED_HEAD=0 \
  bash scripts/serve_dsv41_orthrus.sh
```

如果探针超时，先用 `ip -br addr` 核对两台的地址，用 `ip -4 route get <对端内网 IP>` 检查走哪张网卡。若 SSH 正常但 HCCL 失败，核对两台的 HCCL/RoCE 接口和端口配置，并保留两边的 `docker logs dsv41-probe-node0`、`docker logs dsv41-probe-node1`。下次试跑前，可用 `MODE=probe bash scripts/dsv41_two_node.sh down` 清理失败的探针容器。

固定镜像中已验证离线插件安装、真实 W4A8 权重导出和校验 336 个初始化 head 张量。已有单机八卡 Orthrus 路径验证。内网双机 HCCL、整模逐 token 对照和性能还需在这两台机器上实测。PyPTO 分支的 A3 resident 路径需要另一套 W8A8 权重和单机设备上下文。 [单机验证](ORTHRUS_DSV41_REFERENCE.md#实验性单机-a2-smoke) · [PyPTO 接入](ORTHRUS_DSV41_PYPTO.md)
