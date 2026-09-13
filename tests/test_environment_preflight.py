from experiments.shared.environment import environment_issues, required_distributions


def test_family_requirements_are_role_specific() -> None:
    ar = required_distributions("arllm", stage="all")
    dllm = required_distributions("dllm", stage="all")
    assert "transformers>=5.14,<6" in ar
    assert "transformers==4.53.2" in dllm
    assert any(item.startswith("trl") for item in ar)
    assert not any(item.startswith("trl") for item in dllm)


def test_environment_preflight_reports_versions_and_platform_support() -> None:
    installed = {
        "torch": "2.13.0",
        "transformers": "4.53.2",
        "accelerate": "1.10.0",
        "huggingface-hub": "0.35.0",
        "safetensors": "0.8.0",
    }
    assert not environment_issues(
        "dllm",
        stage="inference",
        installed=installed,
    )
    issues = environment_issues(
        "arllm",
        stage="inference",
        components=("vllm",),
        installed=installed,
        platform_name="win32",
    )
    assert any("transformers" in issue for issue in issues)
    assert any("vLLM requires Linux" in issue for issue in issues)


def test_vllm_preflight_accepts_025_and_026_only() -> None:
    installed = {
        "torch": "2.11.0",
        "transformers": "5.14.0",
        "accelerate": "1.14.0",
        "safetensors": "0.8.0",
        "vllm": "0.25.1",
    }
    assert not environment_issues(
        "arllm",
        stage="inference",
        components=("vllm",),
        installed=installed,
        platform_name="linux",
    )

    installed["vllm"] = "0.26.0"
    assert not environment_issues(
        "arllm",
        stage="inference",
        components=("vllm",),
        installed=installed,
        platform_name="linux",
    )

    installed["torch"] = "2.13.0"
    issues = environment_issues(
        "arllm",
        stage="inference",
        components=("vllm",),
        installed=installed,
        platform_name="linux",
    )
    assert any("torch==2.13.0" in issue and "==2.11.0" in issue for issue in issues)

    installed["torch"] = "2.11.0"
    installed["vllm"] = "0.27.0"
    issues = environment_issues(
        "arllm",
        stage="inference",
        components=("vllm",),
        installed=installed,
        platform_name="linux",
    )
    assert any("vllm==0.27.0" in issue and "<0.27" in issue for issue in issues)
