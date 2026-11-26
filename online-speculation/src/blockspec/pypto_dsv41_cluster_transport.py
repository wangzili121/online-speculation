"""Global HCCL groups over node-local PyPTO allocations."""

import ctypes as C

from .dsv41_online_profile import active_profile, timed_stage


ROOT_INFO_BYTES = 4108


class ClusterCollectives:
    """Match global groups while keeping tensor ownership on the local runtime."""

    def __init__(self, runtime, plan, link, *, native=None):
        if native is None:
            from models.deepseek_v4_1_flash_a3 import moe_transport as native

        # The native Serving launcher returns canonical string IDs from configure_devices.
        local_ids = tuple(str(device) for device in runtime.devices)
        expected_ids = tuple(str(device) for device in range(len(plan.local_ranks)))
        if local_ids != expected_ids or link.node != plan.node:
            raise ValueError("Runtime must use the plan's node-local logical device IDs starting at zero")
        self.runtime, self.plan, self.link = runtime, plan, link
        self.memory, self.channel_type, self.check = native.CANNMemory(), native.HCCLBytes, native._check
        self.channels, self.owned = {}, []
        self.library = C.CDLL("libhccl.so", mode=C.RTLD_GLOBAL)
        p, n, i = C.c_void_p, C.c_uint64, C.c_int
        signatures = {
            "HcclGetRootInfo": [p], "HcclCommInitRootInfo": [C.c_uint32, p, C.c_uint32, C.POINTER(p)],
            "HcclCommDestroy": [p], "HcclAlltoAll": [p, n, i, p, n, i, p, p],
            "HcclAllGather": [p, p, n, i, p, p], "HcclAllReduce": [p, p, n, i, i, p, p],
            "HcclReduceScatter": [p, p, n, i, i, p, p],
        }
        for name, signature in signatures.items():
            native._bind(self.library, name, signature)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        first_error = None
        for channel in reversed(self.owned):
            try:
                channel.close()
            except Exception as error:
                first_error = first_error or error
        if exc_type is None and first_error is not None:
            raise first_error

    def _cross_group(self, group, tag):
        local = self.plan.local_ranks
        encoded = None
        if group[0] in local:
            self.memory.select(group[0] - local.start)
            info = C.create_string_buffer(ROOT_INFO_BYTES)
            self.check(self.library.HcclGetRootInfo(info), "HcclGetRootInfo")
            encoded = info.raw.hex()
        peer = self.link.exchange(tag, encoded)
        if (encoded is None) == (peer is None):
            raise ValueError("Cross-host HCCL group needs exactly one root")
        raw = bytes.fromhex(encoded if encoded is not None else peer)
        if len(raw) != ROOT_INFO_BYTES:
            raise ValueError("Invalid HCCL root-info length")

        def initialize(rank):
            global_rank = local[rank]
            if global_rank not in group:
                return None
            self.memory.select(rank)
            info, comm = C.create_string_buffer(raw, ROOT_INFO_BYTES), C.c_void_p()
            self.check(self.library.HcclCommInitRootInfo(len(group), info, group.index(global_rank), C.byref(comm)),
                       "HcclCommInitRootInfo")
            channel = self.channel_type(self.memory, rank, group, comm.value, self.library)
            self.owned.append(channel)
            acl = self.memory.library
            self.check(acl.aclrtGetCurrentContext(C.byref(channel._context)), "aclrtGetCurrentContext")
            self.check(acl.aclrtCreateStream(C.byref(channel.stream)), "aclrtCreateStream")
            return channel

        return {rank: channel for rank, channel in enumerate(self.runtime.map_ranks(initialize)) if channel is not None}

    def _channels(self, axis):
        if axis not in self.channels:
            local = self.plan.local_ranks
            channels = {}
            for index, group in enumerate(self.plan.groups(axis)):
                selected = [rank - local.start for rank in group if rank in local]
                if not selected:
                    continue
                if len(selected) == len(group):
                    created = self.channel_type.create_group(self.memory, selected, _register_owner=self.owned.append)
                    for channel in created.values():
                        channel.group_ranks = list(group)
                else:
                    created = self._cross_group(group, f"communicator/{axis}/{index}")
                channels.update(created)
            self.channels[axis] = channels
        return self.channels[axis]

    def _check_schedule(self, stage, collective, group, signatures):
        ranks = self.plan.local_ranks
        groups = self.plan.groups(group)
        if group == "tp":
            # RankPlan places entire TP partitions on the same host.
            # Only collectives with remote members require peer rendezvous.
            groups = tuple(members for members in groups if members[0] in ranks)
            shapes = {row["rank"]: row["fields"] for row in signatures}
        else:
            with timed_stage(f"{group}_schedule_control"):
                peer = self.link.exchange(f"schedule/{stage}", dict(collective=collective, group=group, packets=signatures))
            if peer["collective"] != collective or peer["group"] != group:
                raise ValueError("Peers entered different HCCL collectives")
            records = signatures + peer["packets"]
            shapes = {row["rank"]: row["fields"] for row in records}
            if len(records) != sum(self.plan.counts) or set(shapes) != set(range(sum(self.plan.counts))):
                raise ValueError("HCCL packet ownership does not cover the global ranks")
        for members in groups:
            if not shapes[members[0]] or any(shapes[rank] != shapes[members[0]] for rank in members):
                raise ValueError(f"HCCL tensor ABI differs in {group} group {members}")

    def exchange_resident(self, stage, transfers, *, collective="all_to_all", group="ep"):
        """Exchange preallocated device tensors in global rank order."""
        ranks = self.plan.local_ranks
        if len(transfers) != len(ranks):
            raise ValueError("Every node-local rank must provide its resident transfers")
        signatures = [{"rank": rank, "fields": [
            [name, list(source.shape), str(source.dtype), list(destination.shape), str(destination.dtype)]
            for source, destination, name in batch
        ]} for rank, batch in zip(ranks, transfers, strict=True)]
        self._check_schedule(stage, collective, group, signatures)
        channels = self._channels(group)
        for rank, batch in enumerate(transfers):
            for source, destination, _ in batch:
                channels[rank]._validate_device_transfer(source, destination, collective)
        if group != "tp":
            with timed_stage(f"{group}_ready_control"):
                self.link.exchange(f"ready/{stage}", None)

        profile = active_profile()
        if profile is not None and profile.current is not None:
            counters = profile.current["counts"]
            counters[f"{group}_{collective}_calls"] += 1
            counters[f"{group}_{collective}_logical_send_bytes"] += sum(
                source.nbytes for batch in transfers for source, _, _ in batch
            )

        def submit(rank):
            channels[rank].exchange_device_batch(transfers[rank], collective)

        with timed_stage(f"{group}_{collective}_device"):
            self.runtime.map_ranks(submit)

    def exchange(self, stage, packets, collective, group, *, write_only_receive=False):
        """Exchange node-local tensor packets in global rank order."""
        import torch

        ranks = self.plan.local_ranks
        if len(packets) != len(ranks):
            raise ValueError("Every node-local rank must provide one collective packet")
        if collective not in ("all_gather", "all_to_all", "reduce_scatter", "all_reduce"):
            raise ValueError(f"Unsupported collective {collective}")
        for packet in packets:
            for fields in packet.values():
                for value in fields.values():
                    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                            or not value.is_contiguous() or value.ndim == 0 or value.numel() == 0):
                        raise ValueError("Collectives require nonempty contiguous resident tensor descriptors")
        profile = active_profile()
        if profile is not None and profile.current is not None:
            counters = profile.current["counts"]
            counters[f"{group}_{collective}_calls"] += 1
            counters[f"{group}_{collective}_logical_send_bytes"] += sum(
                value.numel() * value.element_size()
                for packet in packets for fields in packet.values() for value in fields.values()
            )
        signatures = [{"rank": rank, "fields": [
            [case, field, list(value.shape), str(value.dtype)]
            for case, values in packet.items() for field, value in values.items()
        ]} for rank, packet in zip(ranks, packets, strict=True)]
        self._check_schedule(stage, collective, group, signatures)
        channels = self._channels(group)

        def prepare(rank):
            channel = channels[rank]
            output, transfers = {}, []
            for case, fields in packets[rank].items():
                output[case] = {}
                for field, value in fields.items():
                    shape = list(value.shape)
                    if collective == "all_gather":
                        shape[0] *= channel.world
                    elif collective == "reduce_scatter":
                        shape[0] //= channel.world
                    received = self.runtime.workspace(
                        shape, value.dtype, rank, fill=None if write_only_receive else 0,
                    )
                    source, destination = self.runtime.tensor(value, rank), self.runtime.tensor(received, rank)
                    channel._validate_device_transfer(source, destination, collective)
                    transfers.append((source, destination, f"{stage}/{case}/{field}"))
                    output[case][field] = received
            return output, transfers

        with timed_stage(f"{group}_collective_buffer_prepare"):
            prepared = self.runtime.map_ranks(prepare)
        if group != "tp":
            with timed_stage(f"{group}_ready_control"):
                self.link.exchange(f"ready/{stage}", None)

        def transfer(rank):
            outputs, transfers = prepared[rank]
            channels[rank].exchange_device_batch(transfers, collective)
            if write_only_receive:
                for fields in outputs.values():
                    for value in fields.values():
                        self.runtime.mark_workspace_written(value, rank)
            return outputs

        with timed_stage(f"{group}_{collective}_device"):
            outputs = self.runtime.map_ranks(transfer)
        return outputs, [dict(rank=ranks[rank], group_ranks=channel.group_ranks, activation_staging=False,
                              collective=collective) for rank, channel in sorted(channels.items())]
