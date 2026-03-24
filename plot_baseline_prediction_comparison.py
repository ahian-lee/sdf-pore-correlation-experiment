#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot prediction diagnostics across baselines.")
    parser.add_argument(
        "--cnn_csv",
        default="/opt/data/private/moffusion/sdf_pore_correlation_repo/baseline_runs/3dcnn_pld_lcd/predictions.csv",
    )
    parser.add_argument(
        "--graph_csv",
        default="/opt/data/private/moffusion/sdf_pore_correlation_repo/baseline_runs/graph_pld_lcd/predictions.csv",
    )
    parser.add_argument(
        "--tree_pld_csv",
        default="/opt/data/private/moffusion/outputs/sdf_pore_correlation_limit5000/diagnostics/pld_extra_trees_predictions.csv",
    )
    parser.add_argument(
        "--tree_lcd_csv",
        default="/opt/data/private/moffusion/outputs/sdf_pore_correlation_limit5000/diagnostics/lcd_extra_trees_predictions.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/opt/data/private/moffusion/sdf_pore_correlation_repo/baseline_runs/diagnostic_figures",
    )
    parser.add_argument("--bins", type=int, default=8)
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_model_predictions(path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    if {"PLD_true", "PLD_pred", "PLD_abs_error", "LCD_true", "LCD_pred", "LCD_abs_error"}.issubset(df.columns):
        for target in ["PLD", "LCD"]:
            rows.append(
                pd.DataFrame(
                    {
                        "name": df["name"],
                        "target": target,
                        "y_true": df[f"{target}_true"],
                        "y_pred": df[f"{target}_pred"],
                        "abs_error": df[f"{target}_abs_error"],
                        "model": model_name,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)
    if {"target", "y_true", "y_pred", "abs_error"}.issubset(df.columns):
        out = df[["name", "target", "y_true", "y_pred", "abs_error"]].copy()
        out["model"] = model_name
        return out
    raise ValueError(f"Unsupported prediction file format: {path}")


def scatter_by_target(df: pd.DataFrame, target: str, output_path: str):
    sub = df[df["target"] == target].copy()
    models = sub["model"].unique().tolist()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5), sharex=True, sharey=True)
    if len(models) == 1:
        axes = [axes]
    lo = min(sub["y_true"].min(), sub["y_pred"].min())
    hi = max(sub["y_true"].max(), sub["y_pred"].max())
    for ax, model in zip(axes, models):
        part = sub[sub["model"] == model]
        rho = part["y_true"].corr(part["y_pred"], method="spearman")
        mae = part["abs_error"].mean()
        ax.scatter(part["y_true"], part["y_pred"], s=12, alpha=0.35, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_title(f"{model}\nMAE={mae:.3f}, rho={rho:.3f}")
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
    fig.suptitle(f"{target}: True vs Predicted")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def binned_error(df: pd.DataFrame, target: str, bins: int) -> pd.DataFrame:
    sub = df[df["target"] == target].copy()
    sub["bin"] = pd.qcut(sub["y_true"], q=bins, duplicates="drop")
    stats = (
        sub.groupby(["model", "bin"], observed=True)
        .agg(
            n=("y_true", "size"),
            true_mean=("y_true", "mean"),
            mae=("abs_error", "mean"),
            bias=("y_pred", lambda x: 0.0),
        )
        .reset_index()
    )
    bias_df = (
        sub.groupby(["model", "bin"], observed=True)
        .apply(lambda x: float((x["y_pred"] - x["y_true"]).mean()))
        .reset_index(name="bias")
    )
    stats = stats.drop(columns=["bias"]).merge(bias_df, on=["model", "bin"], how="left")
    stats["bin_label"] = stats["bin"].apply(lambda iv: f"{iv.left:.2f}-{iv.right:.2f}")
    return stats


def plot_binned_metric(stats: pd.DataFrame, target: str, metric: str, output_path: str):
    pivot = stats.pivot(index="bin_label", columns="model", values=metric)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, model in enumerate(pivot.columns):
        ax.bar(x + (idx - (len(pivot.columns) - 1) / 2) * width, pivot[model].values, width=width, label=model)
    if metric == "bias":
        ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_title(f"{target}: {metric.upper()} by True-value Bin")
    ax.set_ylabel(metric.upper())
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    frames = [
        load_model_predictions(args.cnn_csv, "3dcnn"),
        load_model_predictions(args.graph_csv, "graph_mpnn"),
        load_model_predictions(args.tree_pld_csv, "feature_extra_trees"),
        load_model_predictions(args.tree_lcd_csv, "feature_extra_trees"),
    ]
    df = pd.concat(frames, ignore_index=True)

    for target in ["PLD", "LCD"]:
        scatter_by_target(df, target, os.path.join(args.output_dir, f"{target.lower()}_true_vs_pred_all_models.png"))
        stats = binned_error(df, target, args.bins)
        stats.to_csv(os.path.join(args.output_dir, f"{target.lower()}_binned_error_summary.csv"), index=False)
        plot_binned_metric(stats, target, "mae", os.path.join(args.output_dir, f"{target.lower()}_binned_mae_all_models.png"))
        plot_binned_metric(stats, target, "bias", os.path.join(args.output_dir, f"{target.lower()}_binned_bias_all_models.png"))

    print(args.output_dir)


if __name__ == "__main__":
    main()
