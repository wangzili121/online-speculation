"""Bind the pinned 1122 device graphs to one host's slice of an L4 domain."""

import ast
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RankLocalStack:
    """Expose a local tensor stack with the graph's global rank coordinate."""

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
    if len(domains) != 1:
        raise ValueError("Expected one symmetric communication window in the 1122 graph")
    domain, alloc = domains[0]
    if alloc.args or {kw.arg for kw in alloc.keywords} != {"name", "workers", "window_size", "buffers"}:
        raise ValueError("Unsupported communication-window arguments in the 1122 graph")
    world_aliases = {"world_size"} | {
        node.targets[0].id
        for node in domain.body
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
        and isinstance(iterator.args[1], ast.Name)
        and iterator.args[1].id in world_aliases
        and isinstance(iterator.args[0], ast.Constant)
        and iterator.args[0].value == 0
        and isinstance(iterator.args[2], ast.Constant)
        and iterator.args[2].value == 1
    ]
    submits = [
        node for node in ast.walk(entry)
        if isinstance(node, ast.Call) and _is_name(node.func, "_submit_chip")
    ]
    if len(loops) != 1 or len(submits) != 1:
        raise ValueError("Expected one whole-device-graph submit per global rank")
    loop, = loops
    submit, = submits
    if submit not in ast.walk(loop) or len(submit.args) != 5 or not _is_name(submit.args[4], loop.target.id):
        raise ValueError("The generated chip submit must be pinned to the global rank loop")
    loop.iter = ast.copy_location(ast.Name("_rank_to_worker", ast.Load()), loop.iter)
    submit.args[4] = ast.copy_location(
        ast.Subscript(ast.Name("_rank_to_worker", ast.Load()), submit.args[4], ast.Load()), submit.args[4]
    )
    entry.args.kwonlyargs.append(ast.arg("_rank_to_worker"))
    entry.args.kw_defaults.append(None)
    # Read the exact generated window layout on L4, before running either L3.
    spec = ast.Dict([ast.Constant(kw.arg) for kw in alloc.keywords], [kw.value for kw in alloc.keywords])
    spec = compile(ast.fix_missing_locations(ast.Expression(spec)), str(path), "eval")
    rank_stacks = frozenset(
        source.slice.value
        for node in ast.walk(loop)
        if isinstance(node, ast.Subscript)
        and isinstance(source := node.value, ast.Subscript)
        and _is_name(source.value, "tensors")
        and isinstance(source.slice, ast.Constant)
        and isinstance(source.slice.value, str)
        and any(_is_name(index, loop.target.id) for index in ast.walk(node.slice))
        and not source.slice.value.startswith("__comm_")
    )
    return compile(ast.fix_missing_locations(tree), str(path), "exec"), entry.name, spec, rank_stacks


class NativeRankGraph:
    """Reuse the generated chip ABI and restrict host submits to local workers."""

    def __init__(self, output_dir: Path, world_size: int):
        path = output_dir / "orchestration" / "host_orch.py"
        code, entry, spec, self.rank_stacks = _bind_host_source(path)
        namespace = {"__file__": str(path), "__name__": "_dsv41_ranked_host_orch"}
        exec(code, namespace)
        self.entry = namespace[entry]
        self.allocate_intermediates = namespace["_alloc_intermediates"]
        self.world_size = world_size
        self.window = eval(spec, namespace, {"world_size": world_size})
        if self.window["workers"] != list(range(world_size)):
            raise ValueError("The device graph must use a single global rank space")
        if any(buffer.load_from_host or buffer.store_to_host for buffer in self.window["buffers"]):
            raise ValueError("The 1122 device graph must use uninitialized communication windows")

    def allocate_domain(self, orch, name, members):
        """Attach the persistent window in global rank order to the L4 dispatch."""
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
        view = orch.get_global_domain(domain_id)
        if any(view[worker].domain_rank != rank or view[worker].domain_size != self.world_size
               for rank, worker in rank_to_worker.items()):
            raise ValueError("The imported window does not match the graph's global ranks")
        contexts = {rank: view[worker] for rank, worker in rank_to_worker.items()}
        bound = dict(tensors)
        for name in self.rank_stacks:
            bound[name] = RankLocalStack(tensors[name], tuple(ranks), self.world_size)
        keepalive.append(bound)
        self.entry(
            orch, None, config, tensors=bound, callables=callables, sub_ids={}, _keep=keepalive,
            world_size=self.world_size, _domain_provider=lambda **_: nullcontext(contexts),
            _rank_to_worker=rank_to_worker,
        )
