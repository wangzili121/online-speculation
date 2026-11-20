# 全新机器的 PyPTO 环境

使用 ARM64 Linux Docker 镜像安装配套 PyPTO、Simpler、PTOAS、PyPTO Lib、Serving 和 Orthrus。宿主机提供 Docker、Ascend 驱动和设备节点。镜像构建不占用 NPU，设备探针只使用指定的一张卡。此镜像负责环境准备；双机完整模型执行还需跨机运行时适配和验收。

环境固定为 PyPTO `72a62d13`、Simpler `6b6b88f`、PTOAS 0.70、Lib `e32726b` 加本仓补丁、Serving `cd81894` 加本仓补丁，Python 3.10 和 CPU Torch 2.10。所用 Lib/Serving 配对覆盖现有测试记录，较新的上游优化提交需要重新配对验证。

固定版 Simpler 在 Release 模式下有两处仅供调试断言使用的变量，会触发 `-Werror`。镜像应用 `runtime-release.patch` 显式标记其用途，不改计算和断言；环境报告记录该补丁的 SHA256。

## 构建并传入内网

在可以访问 GitHub、PyPI、PyTorch wheel 站点和镜像仓库的 ARM64 Linux 机器上，从本仓库根目录执行。

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh build
bash online-speculation/scripts/dsv41_pypto_env.sh doctor
bash online-speculation/scripts/dsv41_pypto_env.sh compile-probe
bash online-speculation/scripts/dsv41_pypto_env.sh save /path/to/orthrus-dsv41-pypto.tar.gz
```

将同一份 tar、其相邻的 `.sha256` 文件以及本代码仓复制到两台内网机器。Docker 镜像包含配对源码和已安装的 `pypto_serving_a3`，内网运行时无需联网安装 Python 包。

两台机器都需能访问完整权重目录，可以各自保存一份，也可以使用共享挂载。容器执行镜像内的固定代码；更新实现时需重新构建并传入镜像。

```bash
bash online-speculation/scripts/dsv41_pypto_env.sh load /path/to/orthrus-dsv41-pypto.tar.gz
bash online-speculation/scripts/dsv41_pypto_env.sh host-info
bash online-speculation/scripts/dsv41_pypto_env.sh doctor
bash online-speculation/scripts/dsv41_pypto_env.sh preflight
```

`doctor` 检查包导入、源码配对和版本，输出容器内能看到的驱动/CANN 信息。`compile-probe` 进一步验证 PyPTO 到设备代码的编译，但不执行 NPU 计算。`preflight` 使用默认的 `/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8`；可用 `TARGET_WEIGHTS=/your/path` 覆盖。这三项无需映射 NPU。若只需检查已下载权重的张量头，在任意 Python 3.10+ 环境中执行下面的命令即可。

```bash
python3 online-speculation/scripts/preflight_dsv41_pypto.py --inspect --output checkpoint-layout.json
```

## 验证设备执行

确认所选卡空闲后执行。卡号使用宿主机的物理编号，脚本在容器中映射为可见设备 0。探针会编译并执行一个加法内核，再与 Torch 的结果精确对照。

```bash
NPU_DEVICE=0 bash online-speculation/scripts/dsv41_pypto_env.sh probe
```

成功报告包含 `"kernel_executed": true`。如需进入同一环境排查，执行 `NPU_DEVICE=0 bash online-speculation/scripts/dsv41_pypto_env.sh shell`。

镜像使用官方 ARM A2/A3 开发镜像的固定 digest，其用户态 CANN 为 9.0.0。官方对该基础镜像给出的已验证驱动为 26.0.rc1。现有机器报告的 `npu-smi 25.5.1` 仍需结合 `host-info` 中的实际 driver/version.info 和设备探针判断兼容性。Docker 镜像提供用户态软件，驱动由宿主机提供。 [基础镜像说明](https://github.com/hw-native-sys/pypto/blob/72a62d13efabbb25d240e35944f68ba5bcc22b7f/docs/en/user/04-dev-container.md)

已完成从固定基础镜像构建、正常容器启动导入、设备代码编译和 75 项 Serving CPU 测试。环境验证不等于模型验收：本镜像的 NPU 探针、真实权重完整模型执行和双机通信仍需分别验收，跨机运行时尚待适配。

扩展 Lib 回归有 15 项通过，另有一项 Prefill 编译条目数断言失败。未应用 Orthrus 补丁的 `e32726b` 基线也复现同一失败：测试预期一个条目，原生实现另外包含有限值检查条目。该问题单独记录，未修改主模型或测试断言。
