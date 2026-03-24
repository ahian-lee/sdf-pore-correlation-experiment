#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PLD-range accuracy for 3D CNN and feature baseline.")
    parser.add_argument(
        "--cnn_csv",
        default="/opt/data/private/moffusion/sdf_pore_correlation_repo/baseline_runs/3dcnn_pld_lcd/predictions.csv",
    )
    parser.add_argument(
        "--tree_csv",
        default="/opt/data/private/moffusion/outputs/sdf_pore_correlation_limit5000/diagnostics/pld_extra_trees_predictions.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/opt/data/private/moffusion/sdf_pore_correlation_repo/baseline_runs/pld_range_compare",
    )
    parser.add_argument("--bins", type=int, default=8)
    return parser.parse_args()


def load_cnn(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "name": df["name"],
            "model": "3dcnn",
            "y_true": df["PLD_true"],
            "y_pred": df["PLD_pred"],
            "abs_error": df["PLD_abs_error"],
        }
    )
    out["signed_error"] = out["y_pred"] - out["y_true"]
    return out


def load_tree(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[["name", "y_true", "y_pred", "abs_error", "signed_error"]].copy()
    out["model"] = "feature_extra_trees"
    return out


def binned_stats(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    pieces = []
    for model, part in df.groupby("model"):
        work = part.copy()
        work["bin"] = pd.qcut(work["y_true"], q=bins, duplicates="drop")
        stats = (
            work.groupby("bin", observed=True)
            .agg(
                n=("y_true", "size"),
                true_mean=("y_true", "mean"),
                mae=("abs_error", "mean"),
                rmse=("signed_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                bias=("signed_error", "mean"),
            )
            .reset_index()
        )
        stats["model"] = model
        stats["bin_label"] = stats["bin"].apply(lambda iv: f"{iv.left:.2f}-{iv.right:.2f}")
        pieces.append(stats)
    return pd.concat(pieces, ignore_index=True)


def plot_scatter(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    models = ["3dcnn", "feature_extra_trees"]
    lo = min(df["y_true"].min(), df["y_pred"].min())
    hi = max(df["y_true"].max(), df["y_pred"].max())
    for ax, model in zip(axes, models):
        part = df[df["model"] == model]
        rho = part["y_true"].corr(part["y_pred"], method="spearman")
        mae = part["abs_error"].mean()
        ax.scatter(part["y_true"], part["y_pred"], s=12, alpha=0.35, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_title(f"{model}\nMAE={mae:.3f}, rho={rho:.3f}")
        ax.set_xlabel("True PLD")
        ax.set_ylabel("Predicted PLD")
    fig.suptitle("PLD: True vs Predicted")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric(stats: pd.DataFrame, metric: str, output_path: str):
    pivot = stats.pivot(index="bin_label", columns="model", values=metric)
    x = np.arange(len(pivot.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, pivot["3dcnn"].values, width=width, label="3dcnn")
    ax.bar(x + width / 2, pivot["feature_extra_trees"].values, width=width, label="feature_extra_trees")
    if metric == "bias":
        ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"PLD {metric.upper()} by True-Value Bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_error_scatter(df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, color in [("3dcnn", "#4C78A8"), ("feature_extra_trees", "#F58518")]:
        part = df[df["model"] == model]
        ax.scatter(part["y_true"], part["abs_error"], s=12, alpha=0.30, edgecolors="none", label=model, color=color)
    ax.set_xlabel("True PLD")
    ax.set_ylabel("Absolute Error")
    ax.set_title("PLD Absolute Error vs True PLD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.concat([load_cnn(args.cnn_csv), load_tree(args.tree_csv)], ignore_index=True)
    stats = binned_stats(df, args.bins)
    stats.to_csv(os.path.join(args.output_dir, "pld_binned_stats.csv"), index=False)

    plot_scatter(df, os.path.join(args.output_dir, "pld_true_vs_pred_3dcnn_vs_features.png"))
    plot_metric(stats, "mae", os.path.join(args.output_dir, "pld_binned_mae_3dcnn_vs_features.png"))
    plot_metric(stats, "bias", os.path.join(args.output_dir, "pld_binned_bias_3dcnn_vs_features.png"))
    plot_error_scatter(df, os.path.join(args.output_dir, "pld_abs_error_scatter_3dcnn_vs_features.png"))
    print(args.output_dir)


if __name__ == "__main__":
    main()
