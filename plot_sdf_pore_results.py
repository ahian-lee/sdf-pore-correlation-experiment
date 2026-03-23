#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SDF-pore correlation experiment results.")
    parser.add_argument(
        "--input_dir",
        default="/opt/data/private/moffusion/outputs/sdf_pore_correlation_limit5000",
    )
    parser.add_argument(
        "--output_dir",
        default="/opt/data/private/moffusion/outputs/sdf_pore_correlation_limit5000/figures",
    )
    parser.add_argument("--scatter_sample", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_feature_heatmap(top_corr: pd.DataFrame, output_path: str) -> None:
    subset = top_corr.groupby("target", as_index=False).head(8).copy()
    pivot = subset.pivot(index="feature", columns="target", values="spearman").fillna(0.0)

    fig_h = max(6, 0.35 * len(pivot) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(pivot.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Top SDF Feature Spearman Correlations")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Spearman")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def sampled_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def save_scatter_grid(merged: pd.DataFrame, output_path: str, sample_n: int, seed: int) -> None:
    plot_df = sampled_df(merged, sample_n, seed)
    pairs = [
        ("ch0_q95", "PLD"),
        ("ch0_q75", "PLD"),
        ("ch0_max", "LCD"),
        ("ch0_q95", "LCD"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    for ax, (x, y) in zip(axes, pairs):
        ax.scatter(plot_df[x], plot_df[y], s=10, alpha=0.35, edgecolors="none")
        rho = plot_df[x].corr(plot_df[y], method="spearman")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x} (rho={rho:.3f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_model_bars(metrics: pd.DataFrame, output_path: str) -> None:
    targets = metrics["target"].unique().tolist()
    models = metrics["model"].unique().tolist()
    x = np.arange(len(targets))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for idx, model in enumerate(models):
        sub = metrics[metrics["model"] == model].set_index("target").loc[targets]
        offset = (idx - 0.5) * width
        axes[0].bar(x + offset, sub["r2"].values, width=width, label=model)
        axes[1].bar(x + offset, sub["spearman"].values, width=width, label=model)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(targets)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Model R2 by Target")
    axes[0].set_ylabel("R2")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(targets)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Model Spearman by Target")
    axes[1].set_ylabel("Spearman")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_target_histograms(merged: pd.DataFrame, output_path: str) -> None:
    targets = ["PLD", "LCD", "VF", "GCD"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, target in zip(axes, targets):
        ax.hist(merged[target].dropna().values, bins=40, color="#4C78A8", alpha=0.85)
        ax.set_title(f"{target} Distribution")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    merged = pd.read_csv(os.path.join(args.input_dir, "merged_sdf_properties.csv"))
    top_corr = pd.read_csv(os.path.join(args.input_dir, "top_correlations.csv"))
    metrics = pd.read_csv(os.path.join(args.input_dir, "model_metrics.csv"))

    save_feature_heatmap(top_corr, os.path.join(args.output_dir, "top_feature_heatmap.png"))
    save_scatter_grid(
        merged, os.path.join(args.output_dir, "pld_lcd_scatter_grid.png"), args.scatter_sample, args.seed
    )
    save_model_bars(metrics, os.path.join(args.output_dir, "model_metrics_bars.png"))
    save_target_histograms(merged, os.path.join(args.output_dir, "target_histograms.png"))

    print(args.output_dir)


if __name__ == "__main__":
    main()
