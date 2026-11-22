# 全新机器的 PyPTO 环境

使用 ARM64 Linux Docker 镜像安装配套 PyPTO、Simpler、PTOAS、PyPTO Lib、Serving 和 Orthrus。宿主机提供 Docker、Ascend 驱动和设备节点。镜像构建不占用 NPU，设备探针只使用指定的一张卡。此镜像负责环境准备；双机完整模型执行还需跨机运行时适配和验收。

环境固定为 PyPTO `72a62d13`、Simpler `6b6b88f`、PTOAS 0.70、Lib `e32726b` 加本仓补丁、Serving `cd81894` 加本仓补丁，Python 3.10 和 CPU Torch 2.10。所用 Lib/Serving 配对覆盖现有测试记录，较新的上游优化提交需要重新配对验证。

固定版 Simpler 在 Release 模式下有两处仅供调试断言使用的变量，会触发 `-Werror`。镜像应用 `runtime-release.patch` 显式标记其用途，不改计算和断言；环境报告记录该补丁的 SHA256。

## 已有离线镜像的复验

更新仓库后，在已通过设备探针且 0 号卡空闲的机器上执行这一条命令。它依次核对镜像配对、默认 W8A8 权重目录和 K32 Attention，完整报告保存在当前目录的 `orthrus-handoff.*`。省略 `NPU_DEVICE` 可在另一台机器只检查环境和权重，对照两机输出的 `identity`。设置 `DRAFT_HEAD` 可同时校验指定 head。

```bash
PYPTO_IMAGE=orthrus-dsv41-pypto:72a62d1-offline NPU_DEVICE=0 \
  bash online-speculation/scripts/dsv41_pypto_env.sh handoff-check
```

若需要为文本链路测试准备初始化 head，可在任一装有相同权重的机器导出一次，再把目录复制给另一台。目标盘需留约 10.3 GB。导出使用调用者的 UID/GID，已用普通账号从公开 W8A8 checkpoint 的实际注意力张量导出全部 40 层。示例使用该公开 tokenizer 的占位 token 129279，训练 head 使用训练时的 mask ID。

```bash
PYPTO_IMAGE=orthrus-dsv41-pypto:72a62d1-offline \
  DRAFT_HEAD=/workspace/models/orthrus-dsv41-w8a8-init \
  bash online-speculation/scripts/dsv41_pypto_env.sh init-head --mask-token-id 129279
```

## 双机通信探针

公司内网的 node7 和 node8 可直接按节点名启动。默认使用离线镜像，node7 选物理卡 1，node8 选物理卡 0。确认卡已空闲，先在 node7 运行第一条，再在一分钟内在 node8 运行第二条。需要换卡时，在对应命令前设置 `NPU_DEVICE`。

```bash
# node7
bash online-speculation/scripts/dsv41_pypto_env.sh fabric-check node7

# node8
bash online-speculation/scripts/dsv41_pypto_env.sh fabric-check node8
```

在其他节点上可用 `fabric-check --rank <0|1> --master-address <rank-0-ip>`，并通过 `HCCL_IF_IP` 指定本机通信地址。

探针会只读挂载宿主机已有的 `/etc/hccn.conf`，每次运行会打印 `CANN logs` 目录。两端打印的 `session` 相同，表示已经交换同一份 HCCL root info。此后 HCCL 使用自己的端口建立通信域。初始化失败时，从两端日志核对 `Entry-HcclCommInitRootInfo`、`rootinfo: host ip` 和首条 HCCL 错误。重试时两端都重新启动。

两端的 `passed: true` 表示跨机两卡的 HCCL AllReduce 和 AlltoAll 传回了预期数据。进一步可用双机调度探针检查 8+8 rank 分配、请求波次以及 TP、DP 和 EP 通信。完整模型服务仍由单进程分配设备、权重和缓存。 [双机调度与通信](ORTHRUS_DSV41_PYPTO_CLUSTER.md)

## 内网安装

将镜像 `orthrus-dsv41-pypto.tar.gz`、代码包 `orthrus-dsv41-pypto-code.tar.gz` 及各自的 `.sha256` 文件复制到两台内网机器的工作目录。Docker 镜像包含配对源码和已安装的 `pypto_serving_a3`；内网无需联网安装 Python 包或克隆依赖仓库。

两台机器都需能访问完整权重目录，可以各自保存一份，也可以使用共享挂载。镜像固定运行时和配对 Lib/Serving，复验和 head 导出会挂载当前检出的 Orthrus 源码。

```bash
sha256sum -c orthrus-dsv41-pypto-code.tar.gz.sha256
tar -xzf orthrus-dsv41-pypto-code.tar.gz
bash online-speculation/scripts/dsv41_pypto_env.sh host-info
bash online-speculation/scripts/dsv41_pypto_env.sh load ./orthrus-dsv41-pypto.tar.gz
bash online-speculation/scripts/dsv41_pypto_env.sh doctor
bash online-speculation/scripts/dsv41_pypto_env.sh preflight
```

`doctor` 检查包导入、源码配对和版本，输出容器内能看到的驱动/CANN 信息。`compile-probe` 进一步验证 PyPTO 到设备代码的编译，但不执行 NPU 计算。`preflight` 使用默认的 `/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8`；可用 `TARGET_WEIGHTS=/your/path` 覆盖。这三项无需映射 NPU。若只需检查已下载权重的张量头，在任意 Python 3.10+ 环境中执行下面的命令即可。

```bash
python3 online-speculation/scripts/preflight_dsv41_pypto.py --inspect --output checkpoint-layout.json
```

## 验证设备执行

确认所选卡空闲后执行。`NPU_DEVICE` 选择并挂载宿主机物理卡。单卡容器中的可见设备编号和传给探针的运行时编号均默认为 0。探针会编译并执行一个加法内核，再与 Torch 的结果精确对照。

```bash
NPU_DEVICE=0 bash online-speculation/scripts/dsv41_pypto_env.sh probe
```

成功报告包含 `"kernel_executed": true`。如需进入同一环境排查，执行 `NPU_DEVICE=0 bash online-speculation/scripts/dsv41_pypto_env.sh shell`。

非 0 号物理卡若在设备初始化时报可见编号非法，可用 `NPU_DEVICE=1 bash online-speculation/scripts/dsv41_pypto_env.sh device-map` 检查。此命令只挂载所选卡，分别检查无可见列表、可见列表为 0、可见列表为所选卡号的方式。找到 `set_device=0` 的结果后，将其中的 `visible` 和 `runtime_device` 分别设成 `PYPTO_VISIBLE_DEVICE` 和 `PYPTO_RUNTIME_DEVICE`。JSON 的 `null` 对应 `PYPTO_VISIBLE_DEVICE=unset`。随后用同一设置运行其他探针。 [CANN 可见设备说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0028.html)

旧镜像若在探针里尝试从 GitHub 获取 `pto-isa`，可从当前镜像制作一个补齐运行时缓存的本地镜像。更新代码后，在仓库根目录执行，无需重新编译工具链或下载依赖：

```bash
docker build --network none -t orthrus-dsv41-pypto:72a62d1-offline -f - . <<'EOF'
FROM orthrus-dsv41-pypto:72a62d1
COPY online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh /tmp/cache-pto-isa.sh
RUN bash /tmp/cache-pto-isa.sh
EOF

PYPTO_IMAGE=orthrus-dsv41-pypto:72a62d1-offline NPU_DEVICE=0 \
  bash online-speculation/scripts/dsv41_pypto_env.sh probe
```

单卡加法探针通过后，可在同一张空闲卡上检查 Orthrus K32 Attention。下面的命令挂载当前检出的源码，逐轮运行 SWA 和压缩历史 Attention，检查输出、LSE 和输入缓存。已有离线镜像可以直接用。命令打印一行 JSON 摘要，完整日志保存在宿主机当前目录下的 `orthrus-attention.*`。

```bash
PYPTO_IMAGE=orthrus-dsv41-pypto:72a62d1-offline NPU_DEVICE=0 \
  bash online-speculation/scripts/dsv41_pypto_env.sh attention-check
```

默认使用 TP4 的单 rank 形状（16 头、token-major）。在 `attention-check` 后加 `--compare-baseline` 可同时跑旧版融合 SWA，查看两行摘要，退出码以新实现为准。加 `--heads 8 --no-serving-layout` 可复测原始的 8 头布局。该检查使用合成张量。摘要中 `passed` 和 `npu_execution` 均为 `true` 表示实卡数值通过。

镜像使用官方 ARM A2/A3 开发镜像的固定 digest，其用户态 CANN 为 9.0.0。官方对该基础镜像给出的已验证驱动为 26.0.rc1。现有机器报告的 `npu-smi 25.5.1` 仍需结合 `host-info` 中的实际 driver/version.info 和设备探针判断兼容性。Docker 镜像提供用户态软件，驱动由宿主机提供。 [基础镜像说明](https://github.com/hw-native-sys/pypto/blob/72a62d13efabbb25d240e35944f68ba5bcc22b7f/docs/en/user/04-dev-container.md)

已完成从固定基础镜像构建、正常容器启动导入、设备代码编译和 75 项 Serving CPU 测试。设备探针、真实权重整模执行和双机通信有各自的验收入口。 [双机调度与通信](ORTHRUS_DSV41_PYPTO_CLUSTER.md)

扩展 Lib 回归有 15 项通过，另有一项 Prefill 编译条目数断言失败。未应用 Orthrus 补丁的 `e32726b` 基线也复现同一失败：测试预期一个条目，原生实现另外包含有限值检查条目。该问题单独记录，未修改主模型或测试断言。

## 外网构建与导出

需要重建镜像时，在可以访问 GitHub、PyPI、PyTorch wheel 站点和镜像仓库的 ARM64 Linux 机器上，从本仓库根目录执行。公司机器直接使用上面的离线导入流程。

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh build
bash online-speculation/scripts/dsv41_pypto_env.sh doctor
bash online-speculation/scripts/dsv41_pypto_env.sh compile-probe
bash online-speculation/scripts/dsv41_pypto_env.sh save /path/to/orthrus-dsv41-pypto.tar.gz
```

代理构建需要 Docker Buildx，可用 `docker buildx version` 检查插件。无代理构建在缺少插件时使用 Docker 的旧构建器。若宿主机需要通过公司代理访问依赖站点，设置 `PYPTO_BUILD_PROXY` 指向获准用于这些站点的代理。构建脚本通过 BuildKit secret 临时传入该值。可以从已配置的环境变量读取，例如：

```bash
PYPTO_BUILD_PROXY="${https_proxy:-${HTTPS_PROXY:-}}" \
  bash online-speculation/scripts/dsv41_pypto_env.sh build
```

如果公司代理运行在 Docker 宿主机的回环地址，构建时还需设置 `PYPTO_BUILD_NETWORK=host`。如果 Git 通过镜像或 URL 重写访问 GitHub，需要先确认相同的访问方式可用于容器构建。

如果代理连接成功但容器报 TLS 证书验证失败，将公司提供的 PEM CA 文件放在构建目录之外，并设置 `PYPTO_BUILD_CA=/path/to/company-ca.pem`。构建会临时将该 CA 与镜像已有的信任链合并，供 Git、curl 和 uv 下载依赖时使用。CA 文件不进入最终镜像。

如果构建环境访问 PyPI 超时，可设置 `PYPTO_BUILD_PYPI` 为公司批准且构建容器可访问的 Python 包索引 URL（以 `/simple/` 结尾）。该索引也用于 PTOAS 安装器解析 Python 依赖。Torch CPU wheel 保留专用索引。
