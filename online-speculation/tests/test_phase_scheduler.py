from blockspec.vllm_ascend.phase_scheduler import StaggeredDraftAdmission


def test_staggered_admission_bounds_batch_and_rotates_waiters():
    scheduler = StaggeredDraftAdmission(
        max_batch=2, length_bucket=100, max_delay_steps=2)
    request_ids = ["a", "b", "c", "d"]
    lengths = [10, 20, 210, 220]

    first = scheduler.select(request_ids, lengths, [0, 1, 2, 3])
    second = scheduler.select(request_ids, lengths, [0, 1, 2, 3])

    assert len(first.selected_indices) == 2
    assert len(first.deferred_indices) == 2
    assert set(second.selected_indices) == set(first.deferred_indices)


def test_staggered_admission_cleans_finished_requests():
    scheduler = StaggeredDraftAdmission(
        max_batch=1, length_bucket=100, max_delay_steps=2)
    scheduler.select(["a", "b"], [10, 20], [0, 1])
    decision = scheduler.select(["b"], [21], [0])

    assert decision.selected_indices == [0]
    assert decision.queue_ages == {"b": 0}
