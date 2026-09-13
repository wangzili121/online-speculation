import json

from blockspec.vllm_ascend.profiling import (set_service_enabled,
                                             write_service_config)


def test_service_profile_config_is_bounded_and_toggleable(tmp_path):
    path = tmp_path / "service.json"
    output = tmp_path / "raw"
    write_service_config(path, output, 10)
    value = json.loads(path.read_text())
    assert value["enable"] == 0
    assert value["timelimit"] == 12
    assert "BatchSchedule" in value["domain"]
    set_service_enabled(path, True)
    assert json.loads(path.read_text())["enable"] == 1
    set_service_enabled(path, False)
    assert json.loads(path.read_text())["enable"] == 0
