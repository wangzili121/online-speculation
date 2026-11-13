#!/usr/bin/env python3
"""Report the target topology and check an optional V4.1 Orthrus head."""

import argparse
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
    args = parser.parse_args()
    if args.head is not None and args.mask_token_id is None:
        parser.error("--head requires --mask-token-id")
    if args.head is None and (args.mask_token_id is not None or args.require_trained):
        parser.error("--mask-token-id and --require-trained require --head")
    try:
        config = read_json(args.model / "config.json")
        topology = V41Topology.from_config(config)
        result = asdict(topology)
        result["layers"] = [asdict(role) | {"mode": role.mode} for role in topology.layers]
        result["head_tensor_count"] = len(head_shapes(config, topology))
        if args.head is not None:
            manifest = validate_head(
                args.model, args.head, mask_token_id=args.mask_token_id,
                require_trained=args.require_trained,
            )
            head_config = read_json(args.head / "config.json")
            if head_config.get("architectures") != ["OrthrusDSV41DraftModel"]:
                raise ContractError("head config must declare OrthrusDSV41DraftModel")
            if head_config.get("num_nextn_predict_layers") != BLOCK_SIZE - 1:
                raise ContractError("head config has an incompatible parallel draft width")
            dflash = head_config.get("dflash_config")
            if (
                not isinstance(dflash, dict)
                or dflash.get("mask_token_id") != args.mask_token_id
                or dflash.get("causal") is not False
                or dflash.get("sample_from_anchor") is not False
            ):
                raise ContractError("head config has an incompatible parallel-drafting policy")
            result["head_artifact_role"] = manifest["artifact_role"]
    except (ContractError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
