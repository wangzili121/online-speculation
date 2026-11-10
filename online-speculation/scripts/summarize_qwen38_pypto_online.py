#!/usr/bin/env python3
"""汇总在线实验，输出逐步 CSV、留出集表格及效果曲线。"""

import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", type=Path, nargs="+")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    reports = [json.loads(path.read_text()) for path in a.reports]
    if not all(item.get("completed") for item in reports):
        p.error("both experiments must finish before comparing results")
    for field in ("initial_head_sha256",):
        if any(item[field] != reports[0][field] for item in reports[1:]):
            p.error(f"different {field}")
    if any(item["dataset"]["tokenized_sha256"] != reports[0]["dataset"]["tokenized_sha256"] for item in reports[1:]):
        p.error("different data or tokenization")
    for field in ("steps", "stride", "replay_blocks", "learning_rate", "warmup_steps", "tokens", "share_ar_prefix"):
        if any(item["arguments"][field] != reports[0]["arguments"][field] for item in reports[1:]):
            p.error(f"different experiment setting: {field}")
    variants = [(item["arguments"]["last_layers"], item["arguments"]["width"]) for item in reports]
    if len(set(variants)) != len(variants):
        p.error("duplicate last-N/width variant")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    updates, evaluations, comparison = [], [], []
    for report in sorted(reports, key=lambda item: (item["arguments"]["width"], item["arguments"]["last_layers"])):
        arm = f"last-{report['arguments']['last_layers']}/K{report['arguments']['width']}"
        for row in report["updates"]:
            updates.append(dict(arm=arm, **row))
        for point in report["evaluations"]:
            for split in ("dev", "test"):
                if point[split]:
                    evaluations.append(dict(arm=arm, step=point["step"], split=split, **point[split]["summary"]))
        initial = report["evaluations"][0]["test"]["summary"]
        final = report["evaluations"][-1]["test"]["summary"]
        net = report["online_net"]["summary"]
        comparison.append(dict(arm=arm, initial_accepted_per_round=initial["accepted_per_round"],
            final_accepted_per_round=final["accepted_per_round"], initial_fixed_speedup=initial["lossless_speedup"],
            final_fixed_speedup=final["lossless_speedup"], online_net_speedup=net["lossless_speedup"],
            training_seconds=report["online_net"]["update_seconds"],
            reached_positions=sum(row["reached_positions"] for row in report["updates"])))
    for name, rows in (("updates", updates), ("heldout", evaluations), ("comparison", comparison)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (a.output_dir / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (a.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    colors = ("#2673B8", "#C16A1B", "#45944D", "#9358A6")
    for index, entry in enumerate(comparison):
        arm, color = entry["arm"], colors[index % len(colors)]
        rows = [row for row in updates if row["arm"] == arm]
        axes[0].plot([row["step"] for row in rows], [row["loss"] for row in rows], label=arm, color=color)
        rows = [row for row in evaluations if row["arm"] == arm and row["split"] == "dev"]
        axes[1].plot([row["step"] for row in rows], [row["accepted_per_round"] for row in rows], "o-", color=color)
        axes[2].plot([row["step"] for row in rows],
                     [row["lossless_speedup"] if row["lossless_speedup"] is not None else float("nan") for row in rows],
                     "o-", color=color)
    for axis, title, ylabel in zip(axes, ("Online training", "Held-out acceptance", "Held-out fixed-head inference"),
                                   ("Reached-prefix TV loss", "Accepted candidates / round", "Lossless speedup over paired AR")):
        axis.set(title=title, xlabel="Update step", ylabel=ylabel)
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False)
    axes[2].axhline(1, color="gray", linestyle="--", linewidth=1)
    fig.savefig(a.output_dir / "online-effect.png", dpi=180)
    fig.savefig(a.output_dir / "online-effect.svg")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
