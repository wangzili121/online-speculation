"""Two-host scheduling and device collectives for node-local A3 resident workers."""

from contextlib import suppress
from dataclasses import dataclass
import hashlib
import hmac
import json
import socket
import struct
import time


@dataclass(frozen=True)
class RankPlan:
    """Place entire TP partitions on each host and span them with EP and DP."""

    counts: tuple[int, int]
    tp: int
    node: int

    def __post_init__(self):
        if (len(self.counts) != 2 or self.node not in (0, 1) or self.tp <= 0
                or any(count < 1 or count % self.tp for count in self.counts)):
            raise ValueError("Two nonempty hosts must each contain complete TP partitions")

    @property
    def local_ranks(self):
        first = sum(self.counts[:self.node])
        return range(first, first + self.counts[self.node])

    def groups(self, axis):
        ep = sum(self.counts)
        return {
            "ep": (tuple(range(ep)),),
            "tp": tuple(tuple(range(start, start + self.tp)) for start in range(0, ep, self.tp)),
            "dp": tuple(tuple(range(lane, ep, self.tp)) for lane in range(self.tp)),
        }[axis]

    def local_contract(self, contract):
        """Project a Serving wave without reassigning its global DP and slot identities."""
        world = sum(self.counts)
        topology = dict(dp=world // self.tp, tp=self.tp, ep=world)
        if any(contract["topology"].get(key) != value for key, value in topology.items()):
            raise ValueError("Serving wave topology differs from the cluster rank plan")
        engines = contract["DP_engines"]
        if [engine["dp_rank"] for engine in engines] != list(range(topology["dp"])):
            raise ValueError("Serving wave must describe every DP, including empty partitions")
        for engine in engines:
            requests = engine["requests"]
            if (engine["active"] != sum(len(row["token_ids"]) for row in requests)
                    or len({row["request_slot"] for row in requests}) != len(requests)):
                raise ValueError("Serving wave token count or request ownership is inconsistent")
        ranks = self.local_ranks
        return dict(contract, DP_engines=engines[ranks.start // self.tp:ranks.stop // self.tp])


class ControlLink:
    """Authenticated, ordered JSON control messages. HBM payloads use HCCL."""

    limit = 1 << 20

    def __init__(self, connection, node, token):
        if len(token) < 16 or node not in (0, 1):
            raise ValueError("Control link requires a per-job token of at least 16 bytes and a valid node")
        self.connection, self.node, self.token = connection, node, token
        self.step = 0

    @classmethod
    def connect(cls, node, address, token, timeout):
        if node not in (0, 1) or len(token) < 16 or timeout <= 0:
            raise ValueError("Invalid node, token or timeout")
        if node == 0:
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.settimeout(timeout)
                listener.bind(address)
                listener.listen(1)
                try:
                    connection, _ = listener.accept()
                except TimeoutError as error:
                    raise ConnectionError(f"No peer reached cluster control {address[0]}:{address[1]}") from error
        else:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    connection = socket.create_connection(address, timeout=min(2, timeout))
                    break
                except OSError as error:
                    if time.monotonic() >= deadline:
                        raise ConnectionError(f"Cannot reach cluster control {address[0]}:{address[1]}") from error
                    time.sleep(0.2)
        connection.settimeout(timeout)
        return cls(connection, node, token)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def _read(self, size):
        result = bytearray()
        while len(result) < size:
            chunk = self.connection.recv(size - len(result))
            if not chunk:
                raise ConnectionError("Peer closed the cluster control connection")
            result.extend(chunk)
        return result

    def _send(self, frame):
        data = json.dumps(frame, separators=(",", ":")).encode()
        if len(data) > self.limit:
            raise ValueError("Cluster control message exceeds the size limit")
        tag = hmac.digest(self.token, data, hashlib.sha256)
        self.connection.sendall(struct.pack("!I", len(data)) + tag + data)

    def _receive(self):
        size, = struct.unpack("!I", self._read(4))
        if not 0 < size <= self.limit:
            raise ValueError("Invalid cluster control frame size")
        tag, data = self._read(32), self._read(size)
        if not hmac.compare_digest(tag, hmac.digest(self.token, data, hashlib.sha256)):
            raise ValueError("Cluster control authentication failed")
        return json.loads(data)

    def exchange(self, phase, payload):
        frame = dict(step=self.step, sender=self.node, phase=phase, payload=payload)
        if self.node == 0:
            self._send(frame)
            peer = self._receive()
        else:
            peer = self._receive()
            self._send(frame)
        if (peer["step"], peer["sender"], peer["phase"]) != (self.step, 1 - self.node, phase):
            raise ValueError("Hosts disagreed on the next cluster operation")
        self.step += 1
        return peer["payload"]


class ClusterDispatcher:
    """Dispatch the same phase on both hosts and advance only after both succeed."""

    def __init__(self, plan, link, *, build_id):
        self.plan, self.link = plan, link
        self.epoch, self.next_phase, self.failed = 0, None, False
        config = dict(counts=list(plan.counts), tp=plan.tp, build_id=build_id)
        if config != link.exchange("configure", config):
            raise ValueError("Cluster hosts disagree on the rank plan or software build")

    def execute(self, phase, local_handler, *, command=None):
        if self.failed:
            raise RuntimeError("Discard the cluster session after a failed dispatch")
        try:
            required = self.next_phase
            if (required is not None and phase != required
                    or required is None and phase not in ("prefill", "decode", "draft", "release")
                    or self.epoch == 0 and phase != "prefill"):
                raise ValueError(f"Unexpected {phase} at cluster epoch {self.epoch}")
            control = dict(epoch=self.epoch, command=command)
            remote = self.link.exchange(f"dispatch/{phase}", control)
            if remote["epoch"] != self.epoch or (self.plan.node == 0 and remote["command"] is not None):
                raise ValueError("Cluster dispatch epoch or command ownership differs")
            payload = command if self.plan.node == 0 else remote["command"]
            detail, local_error = None, None
            try:
                detail = local_handler(payload)
            except Exception as error:
                local_error = error
            status = dict(ok=local_error is None, error=str(local_error) if local_error else None)
            peer = self.link.exchange(f"finish/{phase}", status)
            if local_error is not None:
                raise local_error
            if not peer["ok"]:
                raise RuntimeError(f"Peer failed {phase}: {peer['error']}")
            self.next_phase = {"draft": "verify", "verify": "accept"}.get(phase)
            if phase in ("prefill", "decode", "accept"):
                self.epoch += 1
            return detail
        except BaseException:
            self.failed = True
            with suppress(OSError):
                self.link.connection.shutdown(socket.SHUT_RDWR)
            raise

    def execute_wave(self, phase, local_handler, *, contract=None, collect_tokens=False):
        """Route the wave and optionally merge small token rows in Serving order."""
        def owners(wave):
            return [
                (engine["dp_rank"], row["request_slot"], row["generation"])
                for engine in wave["DP_engines"] for row in engine["requests"]
            ]

        def execute(wave):
            local = self.plan.local_contract(wave)
            result = local_handler(local)
            if not collect_tokens:
                return result
            if any(not isinstance(tokens, list) or len(tokens) > 32 or
                   any(type(token) is not int or token < 0 for token in tokens) for tokens in result):
                raise ValueError("Cluster control can gather only short token ID rows")
            encoded = [[list(owner), tokens] for owner, tokens in zip(owners(local), result, strict=True)]
            peer = self.link.exchange(f"tokens/{phase}", encoded)
            rows = {tuple(owner): tokens for owner, tokens in encoded + peer}
            expected = owners(wave)
            if len(encoded) + len(peer) != len(expected) or set(rows) != set(expected):
                raise ValueError("Token result ownership differs from the scheduled wave")
            return [rows[owner] for owner in expected]

        return self.execute(phase, execute, command=contract)
