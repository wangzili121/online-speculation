#!/usr/bin/env python3
"""Apply the pinned Orthrus overlay to vLLM and vLLM-Ascend sources."""

from __future__ import annotations

import argparse
from pathlib import Path


def require_tokens(path: Path, tokens: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [token for token in tokens if token not in text]
    if missing:
        raise SystemExit(f"DSV4 runtime API guard failed for {path}: {missing}")


def validate_vllm_runtime(root: Path) -> None:
    """Pin the vLLM proposer probability contract used by the Ascend overlay."""
    proposer = root / "vllm/v1/spec_decode/llm_base_proposer.py"
    require_tokens(
        proposer,
        (
            "self._enable_probabilistic_draft_probs = (",
            "def _sample_draft_tokens(",
            "def take_last_draft_probs(",
        ),
    )


def validate_dsv4_runtime(root: Path) -> str:
    """Accept only the audited rc1 or later v0.26 nightly DSV4 API family."""
    dsa = root / "vllm_ascend/attention/dsa_v1.py"
    ops = root / "vllm_ascend/ops/dsa.py"
    device = root / "vllm_ascend/device/device_op.py"
    require_tokens(
        dsa,
        (
            "def _is_w8a8_dynamic(",
            "def _forward_o_proj(",
            "npu_vllm_quant_lightning_indexer",
        ),
    )
    require_tokens(ops, ("def _build_kv_cache(",))
    require_tokens(
        device,
        (
            "def unpack_dsa_forward_kv_cache(",
            "def prepare_dsa_indexer_weights(",
        ),
    )

    split_model = root / "vllm_ascend/models/deepseek_v4/model.py"
    split_indexer = root / "vllm_ascend/models/deepseek_v4/indexer.py"
    split_dspark = root / "vllm_ascend/models/deepseek_v4/dspark.py"
    if split_model.is_file() and split_indexer.is_file() and split_dspark.is_file():
        require_tokens(split_model, ("class DeepseekV4Attention(",))
        require_tokens(split_indexer, ("def _select_topk_serial(", "write_cache: bool"))
        require_tokens(dsa, ("def _get_layer_metadata(", "def _mla_prolog_single_stream("))
        return "v0.26-nightly-split"

    model = root / "vllm_ascend/models/deepseek_v4.py"
    dspark = root / "vllm_ascend/models/deepseek_v4_dspark.py"
    require_tokens(model, ("class DeepseekV4Attention(", "class Indexer("))
    require_tokens(dspark, ("class DSparkDeepseekV4ForCausalLM(",))
    require_tokens(
        dsa,
        (
            "def _require_prefill_metadata(",
            "def _require_decode_metadata(",
            "def rotate_activation(",
        ),
    )
    return "v0.26.0rc1-monolithic"


def replace_once(path: Path, old: str, new: str, dry_run: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return "already applied"
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"version guard failed for {path}: expected one anchor, found {count}"
        )
    if not dry_run:
        path.write_text(text.replace(old, new), encoding="utf-8")
    return "would apply" if dry_run else "applied"


def replace_once_or_present(
    path: Path,
    old: str,
    new: str,
    present_tokens: tuple[str, ...],
    dry_run: bool,
) -> str:
    """Apply an rc1 backport or accept a nightly that already has the feature."""
    text = path.read_text(encoding="utf-8")
    if new in text or all(token in text for token in present_tokens):
        return "already provided"
    return replace_once(path, old, new, dry_run)


def replace_one_of(
    path: Path,
    alternatives: tuple[str, ...],
    new: str,
    dry_run: bool,
) -> str:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return "already applied"
    matches = [old for old in alternatives if text.count(old) == 1]
    if len(matches) != 1:
        counts = {old.splitlines()[0]: text.count(old) for old in alternatives}
        raise SystemExit(f"version guard failed for {path}: anchors={counts}")
    if not dry_run:
        path.write_text(text.replace(matches[0], new), encoding="utf-8")
    return "would apply" if dry_run else "applied"


def replace_variant(
    path: Path,
    alternatives: tuple[tuple[str, str], ...],
    applied_marker: str,
    dry_run: bool,
) -> str:
    """Apply the one exact source-layout variant present in a pinned runtime."""
    text = path.read_text(encoding="utf-8")
    if applied_marker in text:
        return "already applied"
    matches = [(old, new) for old, new in alternatives if text.count(old) == 1]
    if len(matches) != 1:
        counts = {old.splitlines()[0]: text.count(old) for old, _ in alternatives}
        raise SystemExit(f"version guard failed for {path}: variant anchors={counts}")
    old, new = matches[0]
    if not dry_run:
        path.write_text(text.replace(old, new), encoding="utf-8")
    return "would apply" if dry_run else "applied"


def patch_vllm(root: Path, dry_run: bool) -> list[str]:
    path = root / "vllm/config/speculative.py"
    changes = []
    changes.append(
        replace_once(
            path,
            'DSparkModelTypes = Literal["dspark"]\n',
            'DSparkModelTypes = Literal["dspark"]\nOrthrusModelTypes = Literal["orthrus"]\n',
            dry_run,
        )
    )
    changes.append(
        replace_once(
            path,
            "    DSparkModelTypes,\n]\nRejectionSampleMethod",
            "    DSparkModelTypes,\n    OrthrusModelTypes,\n]\nRejectionSampleMethod",
            dry_run,
        )
    )
    changes.append(
        replace_once(
            path,
            '                if self.method in ("eagle", "eagle3", "dflash", "dspark"):\n',
            '                if self.method in ("eagle", "eagle3", "dflash", "dspark", "orthrus"):\n',
            dry_run,
        )
    )
    changes.append(
        replace_once(
            path,
            '                if self.method in ("dflash", "dspark"):\n',
            '                if self.method in ("dflash", "dspark", "orthrus"):\n',
            dry_run,
        )
    )
    changes.append(
        replace_once(
            path,
            '            self.method == "draft_model"\n',
            '            self.method in ("draft_model", "orthrus")\n',
            dry_run,
        )
    )
    changes.append(
        replace_once(
            path,
            '        return self.method == "draft_model"\n',
            '        return self.method in ("draft_model", "orthrus")\n',
            dry_run,
        )
    )
    changes.append(
        replace_once(
            path,
            '                if self.method == "medusa":\n'
            '                    draft_hf_overrides = {"model_type": "medusa"}\n'
            '                else:\n',
            '                if self.method == "medusa":\n'
            '                    draft_hf_overrides = {"model_type": "medusa"}\n'
            '                elif self.method == "orthrus":\n'
            '                    # Keep DSV4 hybrid-attention fields; the default draft\n'
            '                    # override rewrites every deepseek_v4 draft to MTP.\n'
            '                    draft_hf_overrides = {\n'
            '                        "architectures": ["OrthrusDSV4DraftModel"]\n'
            '                    }\n'
            '                else:\n',
            dry_run,
        )
    )
    return changes


def patch_ascend(root: Path, dry_run: bool) -> list[str]:
    validate_dsv4_runtime(root)
    changes = []
    spec_init = root / "vllm_ascend/spec_decode/__init__.py"
    changes.append(
        replace_once(
            spec_init,
            "from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer\n",
            "from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer\n"
            "from blockspec.vllm_ascend.dsv4_proposer import AscendOrthrusProposer\n",
            dry_run,
        )
    )
    changes.append(
        replace_once(
            spec_init,
            '    elif method == "dspark":\n        return AscendDSparkProposer(vllm_config, device, runner)\n',
            '    elif method == "dspark":\n        return AscendDSparkProposer(vllm_config, device, runner)\n'
            '    elif method == "orthrus":\n        return AscendOrthrusProposer(vllm_config, device, runner)\n',
            dry_run,
        )
    )

    runner = root / "vllm_ascend/worker/model_runner_v1.py"
    changes.append(
        replace_once(
            runner,
            "            if (isinstance(attn_module, Attention)\n"
            "                    and (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None):\n",
            "            if (kv_tgt_layer := getattr(\n"
            "                    attn_module, \"kv_sharing_target_layer_name\", None)) is not None:\n",
            dry_run,
        )
    )

    proposer = root / "vllm_ascend/spec_decode/llm_base_proposer.py"
    bind = (
        "        bind_shared_backbone = getattr(self.model, \"bind_shared_backbone\", None)\n"
        "        if bind_shared_backbone is not None:\n"
        "            bind_shared_backbone(target_language_model)\n"
        "            logger.info(\"Bound the Orthrus draft head to the live target backbone\")\n"
    )
    changes.append(
        replace_variant(
            proposer,
            (
                (
                    "        self._maybe_share_lm_head(model)\n"
                    "        # The draft FC, embedding, and lm_head boundaries are now aligned.\n",
                    "        self._maybe_share_lm_head(model)\n" + bind
                    + "        # The draft FC, embedding, and lm_head boundaries are now aligned.\n",
                ),
                (
                    "        self._maybe_share_lm_head(model)\n\n        if (\n",
                    "        self._maybe_share_lm_head(model)\n" + bind + "\n        if (\n",
                ),
                (
                    "        self._maybe_share_lm_head(target_language_model)\n\n        if (\n",
                    "        self._maybe_share_lm_head(target_language_model)\n"
                    + bind
                    + "\n        if (\n",
                ),
            ),
            "bind_shared_backbone = getattr(self.model",
            dry_run,
        )
    )

    changes.append(
        replace_once_or_present(
            proposer,
            "        batch_size = common_attn_metadata.batch_size()\n\n"
            "        if token_indices_to_sample is None:\n",
            "        batch_size = common_attn_metadata.batch_size()\n"
            "        self._last_draft_probs = None\n\n"
            "        if token_indices_to_sample is None:\n",
            (
                "self._last_draft_probs = None",
                "sampling_metadata: SamplingMetadata | None = None",
            ),
            dry_run,
        )
    )
    changes.append(
        replace_once(
            proposer,
            '                "is_prefill": is_prefill_batch,\n'
            "            }\n",
            '                "is_prefill": is_prefill_batch,\n'
            '                "sampling_metadata": sampling_metadata,\n'
            "            }\n",
            dry_run,
        )
    )
    changes.append(
        replace_once_or_present(
            proposer,
            "        num_tokens,\n"
            "        is_prefill=None,\n"
            "    ) -> torch.Tensor:\n",
            "        num_tokens,\n"
            "        is_prefill=None,\n"
            "        sampling_metadata=None,\n"
            "    ) -> torch.Tensor:\n",
            ("sampling_metadata: SamplingMetadata | None = None",),
            dry_run,
        )
    )
    changes.append(
        replace_once_or_present(
            proposer,
            "        sample_hidden_states = last_hidden_states[token_indices_to_sample]\n\n"
            "        if get_ascend_config().enable_reduce_sample:\n",
            "        sample_hidden_states = last_hidden_states[token_indices_to_sample]\n"
            "        draft_probs = None\n\n"
            "        if get_ascend_config().enable_reduce_sample:\n",
            (
                "draft_probs_step0: torch.Tensor | None = None",
                "_sample_draft_from_logits",
            ),
            dry_run,
        )
    )
    changes.append(
        replace_once_or_present(
            proposer,
            "            else:\n"
            "                logits = self.model.compute_logits(sample_hidden_states)\n"
            "                if lmhead_tp_enable():\n"
            "                    logits, token_indices_to_sample = self._align_tensor_and_indices(\n"
            "                        logits,\n"
            "                        num_indices,\n"
            "                        token_indices_to_sample,\n"
            "                        ori_token_indices_to_sample,\n"
            "                        is_logits=True,\n"
            "                    )\n"
            "                draft_token_ids = logits.argmax(dim=-1)\n\n"
            "        # Early exit if there is only one draft token to be generated.\n",
            "            elif self.method == \"orthrus\" and sampling_metadata is not None:\n"
            "                draft_token_ids, draft_probs = self._sample_draft_tokens(\n"
            "                    sample_hidden_states, sampling_metadata\n"
            "                )\n"
            "            else:\n"
            "                logits = self.model.compute_logits(sample_hidden_states)\n"
            "                if lmhead_tp_enable():\n"
            "                    logits, token_indices_to_sample = self._align_tensor_and_indices(\n"
            "                        logits,\n"
            "                        num_indices,\n"
            "                        token_indices_to_sample,\n"
            "                        ori_token_indices_to_sample,\n"
            "                        is_logits=True,\n"
            "                    )\n"
            "                draft_token_ids = logits.argmax(dim=-1)\n\n"
            "        # Early exit if there is only one draft token to be generated.\n",
            (
                "draft_token_ids, draft_probs_step0 = self._sample_draft_from_logits(",
                "sampling_metadata",
            ),
            dry_run,
        )
    )
    changes.append(
        replace_once_or_present(
            proposer,
            "            if self.method == \"dspark\":\n"
            "                return draft_token_ids[:, 1:]\n"
            "            else:\n"
            "                # [batch_size, 1]\n"
            "                return draft_token_ids.view(-1, self.num_speculative_tokens)\n",
            "            if self.method == \"dspark\":\n"
            "                return draft_token_ids[:, 1:]\n"
            "            if self.method == \"orthrus\" and draft_probs is not None:\n"
            "                self._last_draft_probs = draft_probs.view(\n"
            "                    -1, self.num_speculative_tokens, draft_probs.shape[-1]\n"
            "                ).contiguous()\n"
            "            # [batch_size, num_speculative_tokens]\n"
            "            return draft_token_ids.view(-1, self.num_speculative_tokens)\n",
            (
                "if draft_probs_step0 is not None:",
                "self._last_draft_probs = draft_probs_step0.view(",
            ),
            dry_run,
        )
    )

    dspark = root / "vllm_ascend/spec_decode/dspark_proposer.py"
    changes.append(
        replace_once_or_present(
            dspark,
            '        if vllm_config.speculative_config.draft_sample_method == "probabilistic":\n',
            '        if (\n'
            '            vllm_config.speculative_config.method == "dspark"\n'
            '            and vllm_config.speculative_config.draft_sample_method == "probabilistic"\n'
            '        ):\n',
            ("DynamicSpecScheduler",),
            dry_run,
        )
    )

    runner = root / "vllm_ascend/worker/model_runner_v1.py"
    changes.append(
        replace_once(
            runner,
            "            if get_pp_group().world_size > 1 and hasattr(\n"
            "                self.drafter, \"take_last_draft_probs\"\n"
            "            ):\n",
            "            if hasattr(self.drafter, \"take_last_draft_probs\"):\n",
            dry_run,
        )
    )
    changes.append(
        replace_once_or_present(
            runner,
            "        draft_probs = (\n"
            "            self._get_spec_decode_draft_probs(spec_decode_metadata)\n"
            "            if get_pp_group().world_size > 1\n"
            "            else None\n"
            "        )\n",
            "        draft_probs = self._get_spec_decode_draft_probs(\n"
            "            spec_decode_metadata\n"
            "        )\n",
            (
                "draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)",
            ),
            dry_run,
        )
    )

    dsa = root / "vllm_ascend/attention/dsa_v1.py"
    changes.append(
        replace_once(
            dsa,
            "        if self.speculative_config:\n"
            "            spec_token_num = self.speculative_config.num_speculative_tokens\n"
            "            self.spec_slot_mapping = [\n",
            "        if (\n"
            "            self.speculative_config\n"
            "            and self.speculative_config.method != \"orthrus\"\n"
            "        ):\n"
            "            spec_token_num = self.speculative_config.num_speculative_tokens\n"
            "            self.spec_slot_mapping = [\n",
            dry_run,
        )
    )
    return changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, default=Path("/vllm-workspace/vllm"))
    parser.add_argument(
        "--ascend-root", type=Path, default=Path("/vllm-workspace/vllm-ascend")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    roots = (args.vllm_root.resolve(), args.ascend_root.resolve())
    required = (
        roots[0] / "vllm/config/speculative.py",
        roots[1] / "vllm_ascend/spec_decode/__init__.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"runtime source files are missing: {missing}")
    validate_vllm_runtime(roots[0])
    api_family = validate_dsv4_runtime(roots[1])
    changes = patch_vllm(roots[0], args.dry_run) + patch_ascend(roots[1], args.dry_run)
    action = "Dry-run passed" if args.dry_run else "Applied"
    print(f"{action}: {len(changes)} guarded Orthrus edits ({api_family})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
