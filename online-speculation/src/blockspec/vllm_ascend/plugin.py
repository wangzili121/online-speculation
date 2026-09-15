"""Register Orthrus model implementations without trusting remote code."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "OrthrusLM", "blockspec.vllm_ascend.model:OrthrusForCausalLM")
    ModelRegistry.register_model(
        "OrthrusDSV4DraftModel",
        "blockspec.vllm_ascend.dsv4:OrthrusDSV4DraftModel",
    )
