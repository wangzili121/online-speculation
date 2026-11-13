#!/usr/bin/env python3
"""Report the target topology and check an optional V4.1 Orthrus head."""

import argparse
from importlib import metadata
import json
from dataclasses import asdict
from pathlib import Path

from blockspec.vllm_ascend.dsv41_contract import (
    BLOCK_SIZE,
    V41Topology,
    head_shapes,
    validate_head,
)
from blockspec.vllm_ascend.head_contract import ContractError, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--require-trained", action="store_true")
    parser.add_argument("--require-plugin", action="store_true")
    args = parser.parse_args()
    if args.head is None and (args.mask_token_id is not None or args.require_trained):
        parser.error("--mask-token-id and --require-trained require --head")
    try:
        config = read_json(args.model / "config.json")
        topology = V41Topology.from_config(config)
        result = asdict(topology)
        result["layers"] = [asdict(role) | {"mode": role.mode} for role in topology.layers]
        result["head_tensor_count"] = len(head_shapes(config, topology))
        if args.head is not None:
            head_config = read_json(args.head / "config.json")
            mask_token_id = args.mask_token_id
            if mask_token_id is None:
                mask_token_id = head_config.get("mask_token_id")
            if type(mask_token_id) is not int:
                raise ContractError("head config must declare mask_token_id")
            manifest = validate_head(
                args.model, args.head, mask_token_id=mask_token_id,
                require_trained=args.require_trained,
            )
            if head_config.get("architectures") != ["OrthrusDSV41DraftModel"]:
                raise ContractError("head config must declare OrthrusDSV41DraftModel")
            if head_config.get("num_nextn_predict_layers") != BLOCK_SIZE - 1:
                raise ContractError("head config has an incompatible parallel draft width")
            dflash = head_config.get("dflash_config")
            if (
                not isinstance(dflash, dict)
                or dflash.get("mask_token_id") != mask_token_id
                or dflash.get("causal") is not False
                or dflash.get("sample_from_anchor") is not False
            ):
                raise ContractError("head config has an incompatible parallel-drafting policy")
            result["head_artifact_role"] = manifest["artifact_role"]
            result["mask_token_id"] = mask_token_id
        if args.require_plugin and not any(
            entry.name == "blockspec_orthrus"
            for entry in metadata.entry_points(group="vllm.general_plugins")
        ):
            raise ContractError(
                "Orthrus vLLM plugin is not installed; run pip install --no-deps -e ."
            )
    except (ContractError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
