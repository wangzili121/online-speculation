"""Reduce a BF16 vocabulary shard before Orthrus returns greedy tokens."""

PACKET_COLUMNS = 8


def make_local_greedy(capacity, vocabulary, *, tile=6464):
    """Emit one (score, local token ID, all-finite) packet per live row."""
    import pypto.language as pl

    if capacity < 1 or not 0 < tile <= vocabulary < 2**24 or vocabulary % tile or tile % 64:
        raise ValueError("Invalid Orthrus vocabulary reduction geometry")
    columns = PACKET_COLUMNS
    finite_max, expected = 3.4028234663852886e38, float(vocabulary)

    @pl.jit
    def local_greedy(
        logits: pl.Tensor[[capacity, vocabulary], pl.BF16],
        packets: pl.Out[pl.Tensor[[capacity, columns], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(active, name_hint="orthrus_vocab_local_max"):
            offsets = pl.add(pl.neg(pl.cast(pl.arange(0, [1, tile], dtype=pl.INT32), pl.FP32)), tile)
            best = pl.cast(-finite_max, pl.FP32)
            index = pl.cast(0, pl.INT32)
            finite_count = pl.cast(0.0, pl.FP32)
            for column in pl.range(0, vocabulary, tile):
                logits_tile = pl.cast(pl.slice(logits, [1, tile], [token, column]), pl.FP32)
                valid = pl.cmp(pl.abs(logits_tile), finite_max, cmp_type=3)
                counts = pl.row_sum(pl.reshape(valid, [8, tile // 8]))
                spread = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), counts)
                finite_count = finite_count + pl.read(pl.row_sum(pl.transpose(spread, 0, 1)), [0, 0])
                local_max = pl.row_max(pl.reshape(logits_tile, [8, tile // 8]))
                repeated = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), local_max)
                value = pl.read(pl.row_max(pl.transpose(repeated, 0, 1)), [0, 0])
                if value > best:
                    matches = pl.cmp(logits_tile, value, cmp_type=0)
                    positions = pl.reshape(pl.mul(matches, offsets), [8, tile // 8])
                    local_first = pl.row_max(positions)
                    spread_first = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), local_first)
                    first = pl.read(pl.row_max(pl.transpose(spread_first, 0, 1)), [0, 0])
                    best = value
                    index = pl.cast(column + tile, pl.INT32) - pl.cast(first, pl.INT32)
            packet = pl.full([1, columns], dtype=pl.FP32, value=0.0)
            pl.write(packet, [0, 0], best)
            pl.write(packet, [0, 1], pl.cast(index, pl.FP32))
            if finite_count == expected:
                pl.write(packet, [0, 2], 1.0)
            packets[token:token + 1, :] = packet
        return packets

    return local_greedy


def merge_greedy_shards(packets, shard_size):
    """Resolve across contiguous TP shards, selecting the lowest ID on ties."""
    import torch

    if not packets:
        raise ValueError("A greedy result needs at least one vocabulary shard")
    packed = torch.stack(packets)
    if not bool(packed[..., 2].eq(1).all()):
        raise ValueError("Non-finite Orthrus vocabulary logits")
    scores, local_ids = packed[..., 0], packed[..., 1].long()
    if bool((local_ids < 0).any() or (local_ids >= shard_size).any()):
        raise ValueError("Orthrus vocabulary reduction returned an invalid shard index")
    owners = scores.argmax(dim=0, keepdim=True)
    return local_ids.gather(0, owners).squeeze(0) + owners.flatten() * shard_size
