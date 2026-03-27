#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot true-vs-pred scatter for regression outputs.")
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--targets", nargs="+", default=["PLD", "LCD"])
    parser.add_argument("--title_prefix", default="")
    return parser.parse_args()


def plot_target(df: pd.DataFrame, target: str, output_dir: str, title_prefix: str):
    true_col = f"{target}_true"
    pred_col = f"{target}_pred"
    mae_col = f"{target}_abs_error"

    x = df[true_col]
    y = df[pred_col]
    mae = df[mae_col].mean()
    r2 = 1.0 - ((x - y) ** 2).sum() / ((x - x.mean()) ** 2).sum()
    spearman = x.corr(y, method="spearman")

    low = min(x.min(), y.min())
    high = max(x.max(), y.max())
    pad = (high - low) * 0.03

    plt.figure(figsize=(6.5, 6))
    plt.scatter(x, y, s=9, alpha=0.35, edgecolors="none")
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1.5)
    plt.xlim(low - pad, high + pad)
    plt.ylim(low - pad, high + pad)
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    title = f"{title_prefix} {target} True vs Pred".strip()
    plt.title(title)
    plt.text(
        0.03,
        0.97,
        f"MAE={mae:.3f}\nR2={r2:.3f}\nSpearman={spearman:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target.lower()}_true_vs_pred.png"), dpi=200)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.predictions_csv)
    for target in args.targets:
        plot_target(df, target, args.output_dir, args.title_prefix)


if __name__ == "__main__":
    main()
