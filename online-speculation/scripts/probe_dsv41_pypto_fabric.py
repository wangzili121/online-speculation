#!/usr/bin/env python3
"""Check native CANN HCCL transport with one NPU per host."""

import argparse
from contextlib import ExitStack
import ctypes
import json
import os
import socket
import sys
import time


ROOT_INFO_BYTES = 4108  # HcclRootInfo in the paired CANN headers.
HCCL_FP32 = 4
HCCL_UINT8 = 7
HCCL_SUM = 0


def rendezvous(rank, address, get_root_info, wait):
    if rank == 0:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(wait)
            server.bind(address)
            server.listen(1)
            print("HCCL rendezvous ready", flush=True)
            try:
                connection, _ = server.accept()
            except TimeoutError as error:
                raise RuntimeError(f"No peer reached TCP rendezvous {address[0]}:{address[1]}") from error
            with connection:
                connection.settimeout(wait)
                root_info = get_root_info()
                connection.sendall(root_info)
        return root_info

    deadline = time.monotonic() + wait
    while True:
        try:
            connection = socket.create_connection(address, timeout=min(5, wait))
            break
        except OSError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Cannot reach TCP rendezvous {address[0]}:{address[1]}: {error}") from error
            time.sleep(0.5)
    with connection:
        connection.settimeout(wait)
        with connection.makefile("rb") as stream:
            result = stream.read(ROOT_INFO_BYTES)
    if len(result) != ROOT_INFO_BYTES:
        raise RuntimeError("Peer sent incomplete HCCL root info")
    return result


def run(rank, device, address, wait):
    lib = os.environ.get("PYPTO_LIB_ROOT")
    if lib:
        sys.path.insert(0, lib)
    from models.deepseek_v4_1_flash_a3.moe_transport import CANNMemory, _bind, _check

    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", str(max(120, wait)))
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", str(wait))
    memory = CANNMemory()
    memory.select(device)
    hccl = ctypes.CDLL("libhccl.so", mode=ctypes.RTLD_GLOBAL)
    ptr, u32, u64, integer = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int
    _bind(hccl, "HcclGetRootInfo", [ptr])
    _bind(hccl, "HcclCommInitRootInfo", [u32, ptr, u32, ctypes.POINTER(ptr)])
    _bind(hccl, "HcclCommDestroy", [ptr])
    _bind(hccl, "HcclAllReduce", [ptr, ptr, u64, integer, integer, ptr, ptr])
    _bind(hccl, "HcclAlltoAll", [ptr, u64, integer, ptr, u64, integer, ptr, ptr])

    root_info = ctypes.create_string_buffer(ROOT_INFO_BYTES)

    def get_root_info():
        _check(hccl.HcclGetRootInfo(root_info), "HcclGetRootInfo")
        return root_info.raw

    root_info.raw = rendezvous(rank, address, get_root_info, wait)

    with ExitStack() as resources:
        comm, stream, source, target = ptr(), ptr(), ptr(), ptr()
        _check(hccl.HcclCommInitRootInfo(2, root_info, rank, ctypes.byref(comm)), "HcclCommInitRootInfo")
        resources.callback(_check_cleanup, hccl.HcclCommDestroy, comm)
        acl = memory.library
        _check(acl.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream")
        resources.callback(_check_cleanup, acl.aclrtDestroyStream, stream)
        inputs, outputs = (ctypes.c_float * 2)(rank * 2 + 1, rank * 2 + 2), (ctypes.c_float * 2)()
        size = ctypes.sizeof(inputs)
        for buffer in (source, target):
            _check(acl.aclrtMalloc(ctypes.byref(buffer), size, 0), "aclrtMalloc")
            resources.callback(_check_cleanup, acl.aclrtFree, buffer)
        memory.copy(source, ctypes.addressof(inputs), size, 1)
        _check(hccl.HcclAllReduce(source, target, 2, HCCL_FP32, HCCL_SUM, comm, stream), "HcclAllReduce")
        _check(acl.aclrtSynchronizeStream(stream), "AllReduce synchronize")
        memory.copy(ctypes.addressof(outputs), target, size, 2)
        reduced = list(outputs)
        peer_bytes = size // 2
        _check(hccl.HcclAlltoAll(source, peer_bytes, HCCL_UINT8, target, peer_bytes, HCCL_UINT8, comm, stream),
               "HcclAlltoAll")
        _check(acl.aclrtSynchronizeStream(stream), "AlltoAll synchronize")
        memory.copy(ctypes.addressof(outputs), target, size, 2)
        exchanged = list(outputs)
        if reduced != [4, 6] or exchanged != [rank + 1, rank + 3]:
            raise RuntimeError(f"HCCL values differ on rank {rank}: sum {reduced}, exchange {exchanged}")
    return dict(passed=True, rank=rank, device=device, all_reduce=reduced, all_to_all=exchanged)


def _check_cleanup(operation, handle):
    result = operation(handle)
    if result:
        raise RuntimeError(f"{operation.__name__} returned {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", required=True, type=int, choices=(0, 1))
    parser.add_argument("--master-address", required=True, help="IPv4 address of rank 0, reachable from rank 1")
    parser.add_argument("--port", type=int, default=29641)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout < 1 or args.device < 0:
        parser.error("expected a valid port, positive timeout and nonnegative device")
    try:
        socket.inet_pton(socket.AF_INET, args.master_address)
    except OSError:
        parser.error("--master-address must be an IPv4 address")
    try:
        result = run(args.rank, args.device, (args.master_address, args.port), args.timeout)
    except (ImportError, OSError, RuntimeError) as error:
        print(json.dumps(dict(passed=False, rank=args.rank, error=str(error))), flush=True)
        raise SystemExit(1) from None
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
