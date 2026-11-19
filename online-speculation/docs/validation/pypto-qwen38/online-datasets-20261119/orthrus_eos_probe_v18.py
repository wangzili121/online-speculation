"""用真实 draft 前向加受控候选验证 EOS；仅作正确性探针，不作性能结果。"""
import time
import torch
import orthrus_bench_dense_v18 as bench
from blockspec.pypto_qwen38.oneserving_tp2 import OrthrusTP2Runner, rank_weights
from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online


def main(*, guard=None):
    references, checks = {}, []
    counters = {"oracle_eos_candidates": 0}
    real_pair, real_propose = bench.run_pair, OrthrusTP2Runner.propose_many
    real_observe, real_finish = TP2Online.observe_teacher, bench.finish_engine

    def run_pair(engine, pair, *args, **kwargs):
        result = real_pair(engine, pair, *args, **kwargs)
        if engine.runner.output_candidates == 0:
            for text, tokens in zip(pair, result["tokens"], strict=True):
                ids = engine.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
                ids = ids["input_ids"] if hasattr(ids, "keys") else ids
                references[tuple(ids)] = tokens
        return result

    def propose(self, items, *args, **kwargs):
        result = real_propose(self, items, *args, **kwargs)  # 正常执行并捕获真实 hidden/logits。
        for (slot, budget), row in zip(items, result, strict=True):
            assert slot.request.sampling.temperature == 0, "oracle probe is greedy only"
            assert slot.request.stop_token_ids, "oracle probe requires EOS stopping"
            future = references[tuple(slot.request.prompt_ids)][len(slot.request.generated):]
            row[:] = (future + [0] * budget)[:budget]
            counters["oracle_eos_candidates"] += any(t in slot.request.stop_token_ids for t in row)
        return result

    def observe(self, slot, batch_slot, emitted, previous, offset, count, **kwargs):
        state = self.feedback
        real_observe(self, slot, batch_slot, emitted, previous, offset, count, **kwargs)
        if kwargs["selected"] and state is not None and any(
                t in slot.request.stop_token_ids for t in emitted):
            assert emitted[-1] in slot.request.stop_token_ids
            assert state["complete"] and state["valid"] == min(state["candidates"], offset + len(emitted))
            checks.append({"delivered": offset + len(emitted), "supervised": state["valid"]})

    def finish(engine, summary, persist):
        summary.update(diagnostic_only=True, oracle_draft_candidates=True,
                       oracle_note="real draft forward; candidates replaced by AR reference solely to force accepted EOS")
        try:
            if summary.get("completed"):
                runner, trainer = engine.runner, engine.runner.online
                deadline = time.monotonic() + 5
                while runner.requests and time.monotonic() < deadline:
                    time.sleep(.001)
                assert not runner.requests and trainer.pending_update is None
                assert counters["oracle_eos_candidates"] and checks
                assert trainer.published_version == trainer.version > trainer.initial_step
                banks = []
                for rank in range(2):
                    for bank, expected in enumerate(rank_weights(runner.head_directory / f"rank{rank}.safetensors")):
                        if bank not in trainer.publish_banks:
                            continue
                        original = torch.empty_like(expected)
                        runner._worker_call("copy_from", original.data_ptr(),
                            trainer.original[bank].shards[rank].data_ptr, original.nbytes, worker_id=rank)
                        published = torch.empty_like(expected)
                        runner._worker_call("copy_from", published.data_ptr(),
                            trainer.updated[bank].shards[rank].data_ptr, published.nbytes, worker_id=rank)
                        initial_ok = torch.equal(original, expected)
                        published_ok = torch.equal(published, trainer.packed[bank][rank])
                        assert initial_ok and published_ok, (bank, rank)
                        banks.append({"rank": rank, "bank": bank, "initial_unchanged": initial_ok,
                                      "published_matches_host": published_ok})
                summary["device_bank_readback"] = banks
                summary["eos_feedback_checks"] = checks
                summary.update(counters)
        except BaseException as error:
            summary.update(completed=False, error=repr(error), error_chain=bench.exception_chain(error))
            raise
        finally:
            real_finish(engine, summary, persist)

    bench.run_pair, OrthrusTP2Runner.propose_many = run_pair, propose
    TP2Online.observe_teacher, bench.finish_engine = observe, finish
    try:
        bench.main(guard=guard)
    finally:
        bench.run_pair, OrthrusTP2Runner.propose_many = real_pair, real_propose
        TP2Online.observe_teacher, bench.finish_engine = real_observe, real_finish
