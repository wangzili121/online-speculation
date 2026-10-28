"""Request-seeded sampling for request-major parallel draft blocks."""

import torch


def sample_seeded_parallel_draft(
    logits, temperatures, generators, block_width, *, use_fp64_gumbel=False
):
    batch = temperatures.numel()
    if logits.ndim != 2 or logits.shape[0] != batch * block_width:
        raise ValueError("draft logits must contain block_width rows per request")
    greedy = temperatures < 1e-5
    safe_temperature = torch.where(greedy, 1.0, temperatures)
    probabilities = (
        logits.float() / safe_temperature.repeat_interleave(block_width)[:, None]
    ).softmax(-1)
    noise = torch.empty_like(
        probabilities, dtype=torch.float64 if use_fp64_gumbel else torch.float32
    ).view(batch, block_width, -1)
    # A request owns all K-1 rows, not just one flattened row. Sharing its
    # generator with verification also keeps draw order independent of batching.
    for request in range(batch):
        noise[request].exponential_(generator=generators.get(request))
    ids = (probabilities / noise.view_as(probabilities)).argmax(-1)
    ids = torch.where(
        greedy.repeat_interleave(block_width), probabilities.argmax(-1), ids
    )
    return ids, probabilities
