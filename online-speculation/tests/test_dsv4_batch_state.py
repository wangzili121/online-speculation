import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("temperatures,counts,cached,has_probs,missing,stale", [
    ([0.0], [31], ["old"], False, 0, 1),
    ([0.8], [31], ["old"], False, 1, 0),
    ([0.8], [31], ["r0"], True, 0, 0),
    ([0.0, 0.8], [31, 0], ["old"], False, 0, 0),
    ([0.8], [0], [], False, 0, 0),
])
def test_probe_distinguishes_stale_greedy_from_missing_random(
        tmp_path, temperatures, counts, cached, has_probs, missing, stale):
    path = Path(__file__).resolve().parents[1] / "scripts/check_dsv4_batch_state.py"
    spec = importlib.util.spec_from_file_location("batch_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = [f"r{i}" for i in range(len(temperatures))]
    probabilities = SimpleNamespace(shape=(sum(counts), 100)) if has_probs else None
    runner = SimpleNamespace(
        use_async_scheduling=False, _draft_probs=probabilities,
        _draft_prob_req_ids=cached,
        input_batch=SimpleNamespace(req_ids=ids, sampling_metadata=SimpleNamespace(
            all_greedy=all(t == 0 for t in temperatures))),
        requests={req: SimpleNamespace(sampling_params=SimpleNamespace(temperature=t))
                  for req, t in zip(ids, temperatures, strict=True)},
        _get_spec_decode_draft_probs=lambda metadata: probabilities,
        propose_draft_token_ids=lambda: [1],
    )
    worker = module.BatchStateWorker()
    worker.model_runner = runner
    worker.rank = 0
    worker.batch_probe_install(str(tmp_path))
    worker.batch_probe_phase("test")
    assert runner._get_spec_decode_draft_probs(SimpleNamespace(num_draft_tokens=counts)) is probabilities
    assert runner.propose_draft_token_ids() == [1]
    report = worker.batch_probe_report()
    assert report["missing_random_probabilities"] == missing
    assert report["stale_greedy_probability_steps"] == stale
