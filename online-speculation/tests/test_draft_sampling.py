import pytest

torch = pytest.importorskip("torch")

from blockspec.vllm_ascend.draft_sampling import sample_seeded_parallel_draft


def test_seeded_blocks_survive_request_reordering_and_global_rng_changes():
    logits = torch.randn(3, 31, 19)
    temperatures = torch.tensor([0.0, 0.7, 1.2])
    seeds = [7, 17, 27]
    generators = {i: torch.Generator().manual_seed(seed) for i, seed in enumerate(seeds)}
    ids, probs = sample_seeded_parallel_draft(logits.flatten(0, 1), temperatures, generators, 31)
    torch.rand(1000)
    order = torch.tensor([2, 0, 1])
    reordered = {i: torch.Generator().manual_seed(seeds[j]) for i, j in enumerate(order.tolist())}
    new_ids, new_probs = sample_seeded_parallel_draft(
        logits[order].flatten(0, 1), temperatures[order], reordered, 31
    )
    assert torch.equal(new_ids.view(3, 31), ids.view(3, 31)[order])
    torch.testing.assert_close(new_probs.view(3, 31, 19), probs.view(3, 31, 19)[order])
    assert torch.equal(ids[:31], logits[0].argmax(-1))
    torch.testing.assert_close(probs.sum(-1), torch.ones(93))


def test_seeded_blocks_preserve_per_request_generator_progress():
    logits = torch.randn(2, 31, 23)
    generators = {i: torch.Generator().manual_seed(i) for i in range(2)}
    for _ in range(2):
        batched, _ = sample_seeded_parallel_draft(
            logits.flatten(0, 1), torch.ones(2), generators, 31
        )
    for i in range(2):
        generator = torch.Generator().manual_seed(i)
        for _ in range(2):
            single, _ = sample_seeded_parallel_draft(
                logits[i], torch.ones(1), {0: generator}, 31
            )
        assert torch.equal(single, batched.view(2, 31)[i])


def test_seeded_blocks_reject_wrong_row_count():
    with pytest.raises(ValueError, match="rows per request"):
        sample_seeded_parallel_draft(torch.zeros(32, 4), torch.ones(1), {}, 31)
