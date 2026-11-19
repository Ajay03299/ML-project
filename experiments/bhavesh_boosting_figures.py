"""Render the combined three-family curve figure from completed tuning CSVs."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from experiments.bhavesh_boosting import GRIDS, round_parameter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tail", action="store_true", help="late-round detail with independent vertical scales")
    args = parser.parse_args(argv)
    out = args.directory
    info = json.loads((out / "tuning_manifest.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=not args.tail)
    for ax, name in zip(axes, GRIDS):
        table = pd.read_csv(out / "tuning" / f"{name}.csv")
        curves = pd.read_csv(out / "tuning" / f"{name}_rounds.csv")
        best = json.loads((out / "best_params" / f"{name}.json").read_text())
        winner = curves[curves["candidate"] == table.iloc[0]["candidate"]]
        if len(table) != 10 or winner.empty:
            raise ValueError(f"{name} search is incomplete; expected ten candidates.")
        if args.tail:
            winner = winner[winner["round"] >= max(2, int(winner["round"].max() * 0.1))]
        ax.plot(winner["round"], winner["train_MAE"], label="training sample")
        ax.plot(winner["round"], winner["val_MAE"], label="chronological validation")
        ax.axvline(
            best[round_parameter(name)],
            color="black",
            linestyle=":",
            label="selected round",
        )
        ax.set(title=name, xlabel="Boosting rounds", ylabel=f"MAE ({info['unit']})")
    axes[0].legend(fontsize=10)
    detail = "late-round detail; independent MAE scales" if args.tail else "full validation, no test tuning"
    fig.suptitle(f"{info['source']} | {info['horizon_min']} min ahead | {detail}")
    fig.tight_layout()
    filename = "boosting_rounds_tail.png" if args.tail else "boosting_rounds.png"
    fig.savefig(out / filename, dpi=160)
    plt.close(fig)
    print(out / filename)


if __name__ == "__main__":
    main()
