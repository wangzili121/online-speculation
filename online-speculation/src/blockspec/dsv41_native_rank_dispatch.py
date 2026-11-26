"""Bind the pinned 1122 device graphs to one host's slice of an L4 domain."""

import ast
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RankLocalStack:
    """Expose a local tensor stack in its global operand index space.

    For DP-owner stacks an index is a DP replica, even though the local
    shard lives on that replica's first TP worker.
    """

    stack: object
    ranks: tuple[int, ...]
    world_size: int

    @property
    def shape(self):
        return (self.world_size, *self.stack.shape[1:])

    def __getitem__(self, key):
        rank, *slices = key if isinstance(key, tuple) else (key,)
        # NZ codegen folds the rank axis into the first physical tile axis.
        if isinstance(rank, slice):
            width = rank.stop - rank.start
            global_rank = rank.start // width
            local_start = self.ranks.index(global_rank) * width
            return self.stack[(slice(local_start, local_start + width, rank.step), *slices)]
        index = self.ranks.index(rank)
        return self.stack[(index, *slices)] if slices else self.stack[index]


def _is_name(node, value):
    return isinstance(node, ast.Name) and node.id == value


def _marked_entry(tree):
    names = {
        target.value.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(target := node.targets[0], ast.Attribute)
        and target.attr == "_pypto_distributed_entry"
        and isinstance(target.value, ast.Name)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
    }
    entries = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(entries) != 1:
        raise ValueError("Expected one generated PyPTO host entry")
    return entries[0]


def _call_args(nodes, name):
    return [node for node in ast.walk(nodes) if isinstance(node, ast.Call) and _is_name(node.func, name)]


def _bind_host_source(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    entry = _marked_entry(tree)
    domains = [
        (node, item.context_expr)
        for node in entry.body
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(fn := item.context_expr.func, ast.BoolOp)
        and isinstance(fn.op, ast.Or)
        and _is_name(fn.values[0], "_domain_provider")
    ]
    if len(domains) > 1:
        raise ValueError("Expected at most one symmetric communication window in the 1122 graph")
    domain, alloc = domains[0] if domains else (entry, None)
    world_aliases = {"world_size"} | {
        node.targets[0].id
        for node in (*entry.body, *domain.body)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and _is_name(node.value, "world_size")
    }
    loops = [
        node for node in domain.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and isinstance(iterator := node.iter, ast.Call)
        and _is_name(iterator.func, "range")
        and len(iterator.args) == 3
        and not iterator.keywords
        and (isinstance(iterator.args[1], ast.Constant) and type(iterator.args[1].value) is int
             or isinstance(iterator.args[1], ast.Name) and iterator.args[1].id in world_aliases)
        and isinstance(iterator.args[0], ast.Constant)
        and iterator.args[0].value == 0
        and isinstance(iterator.args[2], ast.Constant)
        and iterator.args[2].value == 1
    ]
    submits = _call_args(entry, "_submit_chip")
    if not submits or len(loops) > 1 or any(len(submit.args) != 5 for submit in submits):
        raise ValueError("Unsupported 1122 chip dispatch layout")
    if loops:
        loop, = loops
        if submits != _call_args(loop, "_submit_chip") or any(
            not _is_name(submit.args[4], loop.target.id) for submit in submits
        ):
            raise ValueError("The generated chip submits must be pinned to the global rank loop")
        loop.iter = ast.copy_location(
            ast.Call(ast.Name("_local_ranks_in", ast.Load()),
                     [ast.Name("_rank_to_worker", ast.Load()), loop.iter], []),
            loop.iter,
        )
    else:
        # Loss and teacher snapshots submit directly on a scalar owner. Skip
        # the entire entry on the other host, before touching owner tensors.
        owner = submits[0].args[4]
        if alloc is not None or any(ast.dump(submit.args[4]) != ast.dump(owner) for submit in submits):
            raise ValueError("A rank-local entry requires one explicit owner")
        sources = [owner] if isinstance(owner, ast.Constant) and type(owner.value) is int else []
        if isinstance(owner, ast.Name):
            sources = [
                node.value for node in entry.body
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and _is_name(node.targets[0], owner.id)
            ]
        if len(sources) != 1 or not isinstance(sources[0], (ast.Subscript, ast.Constant)):
            raise ValueError("The generated owner must come from a scalar operand")
        guard = ast.parse("if int(_owner) not in _rank_to_worker: return").body[0]
        guard.test.left.args[0] = sources[0]
        entry.body.insert(0, guard)
    for submit in submits:
        rank = submit.args[4]
        submit.args[4] = ast.copy_location(
            ast.Subscript(ast.Name("_rank_to_worker", ast.Load()), rank, ast.Load()), rank,
        )
    entry.args.kwonlyargs.append(ast.arg("_rank_to_worker"))
    entry.args.kw_defaults.append(None)
    # Read the exact generated window layout on L4, before running either L3.
    spec = None
    if alloc is not None:
        if alloc.args or {kw.arg for kw in alloc.keywords} != {"name", "workers", "window_size", "buffers"}:
            raise ValueError("Unsupported communication-window arguments in the 1122 graph")
        mapping = ast.Dict([ast.Constant(kw.arg) for kw in alloc.keywords], [kw.value for kw in alloc.keywords])
        spec = compile(ast.fix_missing_locations(ast.Expression(mapping)), str(path), "eval")
    rank_stacks = frozenset(
        source.slice.value
        for node in ast.walk(domain)
        if isinstance(node, ast.Subscript)
        and isinstance(source := node.value, ast.Subscript)
        and _is_name(source.value, "tensors")
        and isinstance(source.slice, ast.Constant)
        and isinstance(source.slice.value, str)
        and not source.slice.value.startswith("__comm_")
    )
    return compile(ast.fix_missing_locations(tree), str(path), "exec"), entry.name, spec, rank_stacks


def _local_ranks_in(rank_to_worker, ranks):
    return (rank for rank in ranks if rank in rank_to_worker)


class NativeRankGraph:
    """Reuse the generated chip ABI and restrict host submits to local workers."""

    def __init__(self, output_dir: Path, world_size: int):
        path = output_dir / "orchestration" / "host_orch.py"
        try:
            code, entry, spec, self.rank_stacks = _bind_host_source(path)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        namespace = {
            "__file__": str(path), "__name__": "_dsv41_ranked_host_orch", "_local_ranks_in": _local_ranks_in,
        }
        exec(code, namespace)
        self.entry = namespace[entry]
        self.allocate_intermediates = namespace["_alloc_intermediates"]
        self.world_size = world_size
        self.window = None if spec is None else eval(spec, namespace, {"world_size": world_size})
        if self.window is not None:
            if self.window["workers"] != list(range(world_size)):
                raise ValueError("The device graph must use a single global rank space")
            if any(buffer.load_from_host or buffer.store_to_host for buffer in self.window["buffers"]):
                raise ValueError("The 1122 device graph must use uninitialized communication windows")

    def allocate_domain(self, orch, name, members):
        """Attach the persistent window in global rank order to the L4 dispatch."""
        if self.window is None:
            return None
        members = tuple(members)
        if len(members) != self.world_size:
            raise ValueError("Provide one L4 (node, worker) member per global rank")
        return orch.get_or_allocate_domain(
            name=name, members=members, window_size=self.window["window_size"], buffers=self.window["buffers"],
            retain_after_run=True,
        )

    def host_intermediates(self):
        """Create the generated host scratch before the local L3 worker forks."""
        tensors = {}
        self.allocate_intermediates(tensors, world_size=self.world_size)
        return tensors

    def bind_remote_intermediates(self, worker, ranks):
        """Bind generated scratch to post-fork shared Host buffers on a daemon L3."""
        from simpler_setup.torch_interop import torch_dtype_to_datatype

        tensors = {}
        buffers = []
        for name, scratch in self.host_intermediates().items():
            tensors[name] = {}
            for rank in ranks:
                value = scratch[rank]
                buffer = worker.create_buffer(value.nbytes)
                buffers.append(buffer)
                # The generated intermediates are zero-initialized.
                with buffer.host_view() as view:
                    view[:] = bytes(value.nbytes)
                tensors[name][rank] = buffer.tensor(
                    shapes=tuple(value.shape), dtype=torch_dtype_to_datatype(value.dtype)
                )
        return tensors, buffers

    def submit(self, orch, config, *, domain_id, ranks, tensors, callables, keepalive):
        """Submit on L3 with rank-local stacks and already-bound host scratch."""
        rank_to_worker = {rank: worker for worker, rank in enumerate(ranks)}
        if len(rank_to_worker) != len(ranks) or not rank_to_worker or any(
            rank < 0 or rank >= self.world_size for rank in ranks
        ):
            raise ValueError("Invalid rank assignment for this L3 worker")
        contexts = None
        if self.window is not None:
            view = orch.get_global_domain(domain_id)
            if any(view[worker].domain_rank != rank or view[worker].domain_size != self.world_size
                   for rank, worker in rank_to_worker.items()):
                raise ValueError("The imported window does not match the graph's global ranks")
            contexts = {rank: view[worker] for rank, worker in rank_to_worker.items()}
        bound = dict(tensors)
        for name in self.rank_stacks:
            value = tensors[name]
            if not isinstance(value, RankLocalStack):
                bound[name] = RankLocalStack(value, tuple(ranks), self.world_size)
        keepalive.append(bound)
        self.entry(
            orch, None, config, tensors=bound, callables=callables, sub_ids={}, _keep=keepalive,
            world_size=self.world_size, _domain_provider=lambda **_: nullcontext(contexts),
            _rank_to_worker=rank_to_worker,
        )
