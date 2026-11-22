"""实验监控共用 SSH 连接；不修改用户全局 SSH 配置。"""
import os
from pathlib import Path
import stat


HOSTS = {"159.138.5.111", "189.1.227.167", "159.138.54.61"}


def command(host, remote=("python3 -",), *, control=None):
    if host not in HOSTS or control not in (None, "check"):
        raise ValueError("Unexpected observation destination or control operation")
    # macOS 的 Unix socket 路径长度有限，目录不用长的工作区路径。
    base = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path("/tmp")
    directory = base / f"orthrus350-ssh-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError("SSH control directory must be private and owned by the current user")
    args = ["ssh", "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
            "-o", f"ControlPath={directory}/%C", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=20",
            "-o", "ServerAliveCountMax=3"]
    if control is not None:
        return args + ["-O", control, "wangzili@" + host]
    return args + ["wangzili@" + host, *remote]
