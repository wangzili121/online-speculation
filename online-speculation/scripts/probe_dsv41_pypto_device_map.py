#!/usr/bin/env python3
"""Report which CANN runtime ID maps to the NPU mounted in this container."""

import argparse
import ctypes as C
import json
import os


def probe(physical_device):
    acl = C.CDLL("libascendcl.so", mode=C.RTLD_GLOBAL)
    result = dict(physical_device=physical_device, visible=os.getenv("ASCEND_RT_VISIBLE_DEVICES"))
    result["init"] = acl.aclInit(None)
    count = C.c_uint32()
    try:
        result["count_status"] = acl.aclrtGetDeviceCount(C.byref(count))
        result["count"] = count.value
        if result["init"] == 0 and result["count_status"] == 0:
            logical = C.c_int32(-1)
            result["physical_to_logical"] = acl.aclrtGetLogicDevIdByPhyDevId(physical_device, C.byref(logical))
            if result["physical_to_logical"] != 0:
                return result
            result["logical_device"] = logical.value
            device = C.c_int32(-1)
            result["logical_to_runtime"] = acl.aclrtGetUserDevIdByLogicDevId(logical.value, C.byref(device))
            if result["logical_to_runtime"] == 0:
                result["runtime_device"] = device.value
                result["set_device"] = acl.aclrtSetDevice(device.value)
                if result["set_device"] == 0:
                    result["reset_device"] = acl.aclrtResetDevice(device.value)
    finally:
        if result["init"] == 0:
            acl.aclFinalize()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physical_device", type=int)
    result = probe(parser.parse_args().physical_device)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result.get("set_device") == result.get("reset_device", -1) == 0 else 1)
