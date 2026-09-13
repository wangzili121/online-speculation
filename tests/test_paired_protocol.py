from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path("experiments/shared/paired_protocol.py")
    spec = importlib.util.spec_from_file_location("paired_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_llada_protocol_covers_every_declared_ar_experiment_family():
    module = _module()
    _, sections = module.load_pairing(Path("configs/gsm8k_llada_moe_3090.toml"))

    assert set(sections) == set(module.EXPECTED_SETS)
    assert {pair.ar for pair in sections["main_pairs"]} == module.AR_MAIN_METHODS
    assert all(pair.dllm for pairs in sections.values() for pair in pairs)


def test_main_pairing_marks_exact_and_adapted_relations_explicitly():
    module = _module()
    _, sections = module.load_pairing(Path("configs/gsm8k_llada_moe_3090.toml"))
    relations = {pair.relation for pair in sections["main_pairs"]}

    assert relations == {"exact_rule", "matched_role", "adapted"}


def test_frozen_sets_stay_synchronized_with_arllm_experiment_sources():
    module = _module()
    from experiments.arllm.gsm8k_distribution_audit import DEFAULT_METHODS
    from experiments.arllm.gsm8k_is_passk import IS_PASSK_METHODS
    from experiments.arllm.gsm8k_passk import PASSK_METHODS
    from experiments.shared.methods import AR_PAIRED_METHODS
    from experiments.arllm.summarize_gsm8k_dynamic_is import (
        METHODS as DYNAMIC_METHODS,
    )

    assert module.AR_MAIN_METHODS == set(AR_PAIRED_METHODS)
    assert module.AR_PASSK_METHODS == set(PASSK_METHODS) | set(IS_PASSK_METHODS)
    assert module.AR_DISTRIBUTION_METHODS == set(DEFAULT_METHODS)
    assert module.AR_DYNAMIC_ARMS == set(DYNAMIC_METHODS)
