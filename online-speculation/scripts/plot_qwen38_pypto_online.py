#!/usr/bin/env python3
"""绘制真实在线更新 loss、固定 dev 接受长度和计入更新的吞吐。"""

import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reports = [json.loads(path.read_text()) for path in a.reports]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for report in reports:
        n = report["arguments"]["last_layers"]
        label = f"last-{n}"
        updates = report["updates"]
        steps = [row["step"] for row in updates]
        smoothed = [statistics.mean(row["loss"] for row in updates[max(0, i-7):i+1])
                    for i in range(len(updates))]
        axes[0].plot(steps, smoothed, label=label)
        dev = [row for row in report["evaluations"] if row.get("dev")]
        axes[1].plot([row["step"] for row in dev],
            [row["dev"]["summary"]["accepted_per_round"] for row in dev], "o-", label=label)
        if report.get("online_net", {}).get("versus_frozen_head_ratio") is not None:
            axes[2].bar(label, report["online_net"]["versus_frozen_head_ratio"], width=.5)
    axes[0].set(title="Online TV (8-update moving mean)", xlabel="Optimizer updates", ylabel="TV loss")
    axes[1].set(title="Held-out dev; no learning during evaluation", xlabel="Optimizer updates",
                ylabel="Accepted candidates / round")
    axes[2].set(title="Serving throughput including updates", ylabel="Online / frozen initial head")
    axes[2].axhline(1, color="gray", linestyle="--", linewidth=1)
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
    for ax in axes[:2]:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
