import torch

from blockspec.pypto_dsv41_verify_cache import VerifyIndexKeys


def test_verifier_shares_keys_only_within_each_source_request_and_live_page_map():
    producer = dict(mode="full", kv_source=20, physical_layer=20, ratio=1)
    reader = dict(mode="reindex", kv_source=20, physical_layer=36, ratio=1)
    cache = VerifyIndexKeys((producer, reader))
    table = torch.tensor([7, 8, -1])
    builds = []

    def gather():
        pair = torch.tensor([len(builds)]), torch.tensor([0.5])
        builds.append(pair)
        return pair

    initial = cache.get(producer, 8, "fresh", 0, 129, table, gather)
    assert initial is cache.get(reader, 8, "fresh", 0, 129, torch.tensor([7, 8, 9]), gather)
    assert initial is not cache.get(reader, 8, "fresh", 1, 129, table, gather)
    assert initial is not cache.get(reader, 8, "continuation", 0, 129, table, gather)
    assert initial is not cache.get(reader, 9, "fresh", 0, 129, table, gather)
    remapped = cache.get(reader, 8, "fresh", 0, 129, torch.tensor([8, 7, -1]), gather)
    assert initial is not remapped
    shortened = cache.get(reader, 8, "fresh", 0, 128, torch.tensor([8, 7, -1]), gather)
    assert remapped is not shortened
    retained = cache.finish_layer(35, {8: 0, 9: 1})
    assert [local for local, _ in retained] == [0, 0, 0, 0, 0, 0, 1, 1]
    assert [id(value) for _, value in retained][:2] == [id(value) for value in shortened]
    assert cache.finish_layer(36, {8: 0, 9: 1}) == ()
    assert not cache.packed
