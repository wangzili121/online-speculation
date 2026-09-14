import json

from blockspec.vllm_ascend.profiling import (set_service_enabled,
                                             write_service_config)


def test_service_profile_config_has_safe_limit_and_is_toggleable(tmp_path):
    path = tmp_path / "service.json"
    output = tmp_path / "raw"
    write_service_config(path, output, 10)
    value = json.loads(path.read_text())
    assert value["enable"] == 0
    assert value["timelimit"] == 120
    assert value["domain"] == ""
    set_service_enabled(path, True)
    assert json.loads(path.read_text())["enable"] == 1
    set_service_enabled(path, False)
    assert json.loads(path.read_text())["enable"] == 0


def test_service_startup_profile_runs_until_server_exit(tmp_path):
    path = tmp_path / "service.json"
    write_service_config(path, tmp_path / "raw", 30, enabled=True)
    value = json.loads(path.read_text())
    assert value["enable"] == 1
    assert value["timelimit"] == 0
