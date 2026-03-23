#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Prediction diagnostics for SDF-pore experiment.")
    parser.add_argument(
        "--input_csv",
        default="/opt/data/private/moffusion/outputs/sdf_pore_correlation_limit5000/merged_sdf_properties.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/opt/data/private/moffusion/outputs/sdf_pore_correlation_limit5000/diagnostics",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--model", choices=["ridge", "extra_trees"], default="extra_trees")
    parser.add_argument("--bins", type=int, default=8)
    return parser.parse_args()


def build_model(name: str, seed: int):
    if name == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=RidgeCV(alphas=np.logspace(-3, 3, 13)),
                        transformer=StandardScaler(),
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=300,
                    random_state=seed,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def fit_predict(df: pd.DataFrame, target: str, model_name: str, seed: int, test_size: float) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in {"name", "PLD", "LCD", "VF", "GCD"}]
    work = df[["name", target] + feature_cols].dropna(subset=[target]).copy()
    X = work[feature_cols]
    y = work[target]
    names = work["name"]

    X_train, X_test, y_train, y_test, n_train, n_test = train_test_split(
        X, y, names, test_size=test_size, random_state=seed
    )
    model = build_model(model_name, seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    out = pd.DataFrame(
        {
            "name": n_test.to_numpy(),
            "target": target,
            "y_true": y_test.to_numpy(),
            "y_pred": pred,
        }
    )
    out["abs_error"] = np.abs(out["y_pred"] - out["y_true"])
    out["signed_error"] = out["y_pred"] - out["y_true"]
    out.attrs["r2"] = float(r2_score(y_test, pred))
    out.attrs["mae"] = float(mean_absolute_error(y_test, pred))
    out.attrs["spearman"] = float(pd.Series(pred).corr(pd.Series(y_test).reset_index(drop=True), method="spearman"))
    return out


def make_scatter(df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(df["y_true"], df["y_pred"], s=12, alpha=0.35, edgecolors="none")
    lo = min(df["y_true"].min(), df["y_pred"].min())
    hi = max(df["y_true"].max(), df["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(
        f"{df['target'].iloc[0]} True vs Predicted\n"
        f"R2={df.attrs['r2']:.3f}, MAE={df.attrs['mae']:.3f}, Spearman={df.attrs['spearman']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_binned_error(df: pd.DataFrame, bins: int, output_path: str) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["bin"] = pd.qcut(plot_df["y_true"], q=bins, duplicates="drop")
    stats = (
        plot_df.groupby("bin", observed=True)
        .agg(
            n=("y_true", "size"),
            true_mean=("y_true", "mean"),
            mae=("abs_error", "mean"),
            rmse=("signed_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            bias=("signed_error", "mean"),
        )
        .reset_index()
    )

    labels = [f"{iv.left:.2f}-{iv.right:.2f}" for iv in stats["bin"]]
    x = np.arange(len(stats))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(x, stats["mae"], color="#4C78A8", alpha=0.9)
    axes[0].set_ylabel("MAE")
    axes[0].set_title(f"{df['target'].iloc[0]} Error by True-Value Bin")

    axes[1].bar(x, stats["bias"], color="#F58518", alpha=0.9)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Bias")
    axes[1].set_xlabel("True-value bins")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    stats["bin_label"] = labels
    return stats


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    all_stats = []
    for target in ["PLD", "LCD"]:
        pred_df = fit_predict(df, target, args.model, args.seed, args.test_size)
        pred_df.to_csv(os.path.join(args.output_dir, f"{target.lower()}_{args.model}_predictions.csv"), index=False)
        make_scatter(pred_df, os.path.join(args.output_dir, f"{target.lower()}_{args.model}_true_vs_pred.png"))
        stats = make_binned_error(
            pred_df, args.bins, os.path.join(args.output_dir, f"{target.lower()}_{args.model}_binned_error.png")
        )
        stats.insert(0, "target", target)
        all_stats.append(stats)

    pd.concat(all_stats, ignore_index=True).to_csv(
        os.path.join(args.output_dir, f"{args.model}_binned_error_summary.csv"), index=False
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
