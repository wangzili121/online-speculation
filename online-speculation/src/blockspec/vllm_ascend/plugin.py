"""Register the Orthrus checkpoint architecture without trusting remote code."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "OrthrusLM", "blockspec.vllm_ascend.model:OrthrusForCausalLM")
