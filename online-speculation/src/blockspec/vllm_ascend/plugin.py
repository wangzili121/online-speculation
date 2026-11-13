"""Register Orthrus model implementations without trusting remote code."""


def _install_dsv41_speculator_dispatch():
    """Add one architecture dispatch missing from pinned Ascend MRV2.

    vLLM 0.32 validates parallel draft models under ``method=dflash``.  The
    architecture, rather than that scheduler-family label, selects Orthrus.
    Patching both module bindings keeps this a plugin-scoped compatibility
    hook; no vLLM or vLLM-Ascend source tree is edited.
    """
    try:
        import vllm_ascend.worker.v2.model_runner as model_runner
        import vllm_ascend.worker.v2.spec_decode as spec_decode
    except ModuleNotFoundError as exc:
        # Model registration is also useful to tools that only install vLLM.
        # The runtime dispatch is required only when the Ascend worker exists.
        if exc.name == "vllm_ascend" or (exc.name or "").startswith("vllm_ascend."):
            return
        raise

    current = spec_decode.init_speculator
    if getattr(current, "_orthrus_dsv41_dispatch", False):
        return

    def init_speculator(vllm_config, device, req_states=None):
        speculative = vllm_config.speculative_config
        architectures = (
            set(speculative.draft_model_config.architectures)
            if speculative is not None and speculative.draft_model_config is not None
            else set()
        )
        if architectures & {
            "OrthrusDSV41DraftModel",
            "DFlashOrthrusDSV41DraftModel",
        }:
            from .dsv41_speculator import AscendOrthrusV41Speculator

            return AscendOrthrusV41Speculator(vllm_config, device)
        return current(vllm_config, device, req_states)

    init_speculator._orthrus_dsv41_dispatch = True
    spec_decode.init_speculator = init_speculator
    model_runner.init_speculator = init_speculator


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "OrthrusLM", "blockspec.vllm_ascend.model:OrthrusForCausalLM")
    ModelRegistry.register_model(
        "OrthrusDSV4DraftModel",
        "blockspec.vllm_ascend.dsv4:OrthrusDSV4DraftModel",
    )
    ModelRegistry.register_model(
        "OrthrusDSV41DraftModel",
        "blockspec.vllm_ascend.dsv41_model:OrthrusDSV41DraftModel",
    )
    # vLLM 0.32 prefixes the draft architecture when method=dflash.  The alias
    # only grants access to that parallel scheduling lifecycle; both names
    # resolve to the same Orthrus model implementation.
    ModelRegistry.register_model(
        "DFlashOrthrusDSV41DraftModel",
        "blockspec.vllm_ascend.dsv41_model:OrthrusDSV41DraftModel",
    )
    _install_dsv41_speculator_dispatch()
