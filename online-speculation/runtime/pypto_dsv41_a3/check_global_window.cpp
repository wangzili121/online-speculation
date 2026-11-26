// Probe the physical-memory export used by Simpler's A3 global domain.
#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

constexpr std::size_t requested_bytes = 4096;

struct Probe {
    const char *stage = "granularity";
    aclError status = ACL_SUCCESS;
    aclError cleanup_status = ACL_SUCCESS;
    std::size_t bytes = 0;
};

void remember_error(aclError status, aclError &first_error) {
    if (first_error == ACL_SUCCESS) {
        first_error = status;
    }
}

Probe export_window(int device, bool fabric) {
    Probe result;
    aclrtPhysicalMemProp properties{};
    properties.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    properties.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = static_cast<std::uint32_t>(device);
    properties.memAttr = fabric ? ACL_HBM_MEM_HUGE : ACL_HBM_MEM_NORMAL;
    std::size_t granularity = 0;
    result.status = aclrtMemGetAllocationGranularity(&properties, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity);
    if (result.status != ACL_SUCCESS) {
        return result;
    }
    if (granularity == 0) {
        result.status = ACL_ERROR_INVALID_PARAM;
        return result;
    }
    result.bytes = ((requested_bytes + granularity - 1) / granularity) * granularity;

    result.stage = "allocate";
    aclrtDrvMemHandle allocation = nullptr;
    result.status = aclrtMallocPhysical(&allocation, result.bytes, &properties, 0);
    if (result.status != ACL_SUCCESS) {
        return result;
    }

    result.stage = "reserve";
    void *base = nullptr;
    result.status = aclrtReserveMemAddress(&base, result.bytes, 0, nullptr, fabric ? HUGE_PAGE_TYPE : 0);
    bool mapped = false;
    if (result.status == ACL_SUCCESS) {
        result.stage = "map";
        result.status = aclrtMapMem(base, result.bytes, 0, allocation, 0);
        mapped = result.status == ACL_SUCCESS;
    }
    if (mapped) {
        result.stage = "export";
        if (fabric) {
            aclrtMemFabricHandle handle{};
            result.status = aclrtMemExportToShareableHandleV2(
                allocation, ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, &handle
            );
        } else {
            std::uint64_t handle = 0;
            result.status = aclrtMemExportToShareableHandleV2(
                allocation, ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, &handle
            );
        }
        if (result.status == ACL_SUCCESS) {
            result.stage = "zero";
            result.status = aclrtMemset(base, result.bytes, 0, result.bytes);
        }
        remember_error(aclrtUnmapMem(base), result.cleanup_status);
    }
    if (base != nullptr) {
        remember_error(aclrtReleaseMemAddress(base), result.cleanup_status);
    }
    remember_error(aclrtFreePhysical(allocation), result.cleanup_status);
    return result;
}

void print_probe(const Probe &result) {
    std::cout << "{\"stage\":\"" << result.stage << "\",\"status\":" << result.status
              << ",\"cleanup_status\":" << result.cleanup_status << ",\"bytes\":" << result.bytes << '}';
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    const int device = std::atoi(argv[1]);
    const aclError initialized = aclInit(nullptr);
    const aclError selected = initialized == ACL_SUCCESS ? aclrtSetDevice(device) : initialized;
    if (selected != ACL_SUCCESS) {
        std::cout << "{\"scope\":\"local-window-export\",\"device\":" << device
                  << ",\"init\":" << initialized << ",\"select\":" << selected << "}\n";
        if (initialized == ACL_SUCCESS) {
            aclFinalize();
        }
        return 2;
    }

    const char *soc = aclrtGetSocName();
    const Probe local = export_window(device, false);
    const Probe fabric = export_window(device, true);
    std::cout << "{\"scope\":\"local-window-export\",\"device\":" << device
              << ",\"soc\":\"" << (soc == nullptr ? "unknown" : soc) << "\",\"local_ipc\":";
    print_probe(local);
    std::cout << ",\"a3_fabric\":";
    print_probe(fabric);
    const aclError reset = aclrtResetDevice(device);
    const aclError finalized = aclFinalize();
    std::cout << ",\"reset\":" << reset << ",\"finalize\":" << finalized << "}\n";
    if (fabric.cleanup_status != ACL_SUCCESS || local.cleanup_status != ACL_SUCCESS || reset != ACL_SUCCESS ||
        finalized != ACL_SUCCESS) {
        return 2;
    }
    return fabric.status == ACL_SUCCESS ? 0 : 1;
}
