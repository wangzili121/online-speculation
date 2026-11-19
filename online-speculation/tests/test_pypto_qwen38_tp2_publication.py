import pytest

from blockspec.pypto_qwen38.tp2_publication import publication_plan
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
from blockspec.pypto_qwen38.training_flat import FlatLayout


@pytest.mark.parametrize("scope,banks", [("full", set(range(9, 15))),
                                       ("norms", {12, 13}), ("output_norms", {12, 13, 14})])
def test_real_layout_contains_exactly_the_published_banks(scope, banks):
    layout = FlatLayout(SuffixGraph(1, parameter_scope=scope).parameters)
    plan = publication_plan(layout, banks)
    assert {row[0] for row in plan} == banks
    for bank, _, source, target, dtype, mode in plan:
        if bank in (12, 13):
            assert source == (1, 256) and target == (2, 256)
            assert dtype == "fp32" and mode == "replicate"
        elif bank == 14:
            assert source == (5120, 6144) and target == (10240, 3072)
            assert mode == "columns"
        else:
            assert source == target and mode == "rows"
    with pytest.raises(ValueError, match="bank"):
        publication_plan(layout, banks | {8})


def test_partial_gdn_publication_is_not_silently_accepted():
    layout = FlatLayout(SuffixGraph(4).parameters)
    with pytest.raises(ValueError, match="last-1"):
        publication_plan(layout, set(range(15)))


def test_dma_keeps_the_owner_allocation_and_passes_a_separate_offset():
    from types import SimpleNamespace
    from blockspec.pypto_qwen38.tp2_publication import PublicationPrograms
    layout = FlatLayout(SuffixGraph(1).parameters)
    obj = object.__new__(PublicationPrograms)
    obj.plan = publication_plan(layout, set(range(9, 15)))
    destinations = {i: SimpleNamespace(data_ptr=10000 + i * 4096) for i in (12, 13, 14)}
    source = SimpleNamespace(data_ptr=4096)
    mapping = obj.sources(layout, source, destinations)
    entries = {entry.parameter.name: entry for entry in layout.entries}
    for bank, name, *_, mode in obj.plan:
        pointer, offset = mapping[bank]
        if mode == "rows":
            assert pointer == source.data_ptr
            assert offset == entries[name].offset * 2
        else:
            assert pointer == destinations[bank].data_ptr and offset == 0
