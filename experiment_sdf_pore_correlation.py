#!/usr/bin/env python
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

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


DEFAULT_TARGETS = ["PLD", "LCD", "VF", "GCD"]


@dataclass
class MatchInfo:
    matched_names: List[str]
    sdf_count: int
    property_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze correlation between SDF npy fields and pore properties."
    )
    parser.add_argument(
        "--sdf_dir",
        default="/opt/data/private/moffusion/autofusion/data/resolution_32",
    )
    parser.add_argument(
        "--property_csv",
        default="/opt/data/private/moffusion/data/properties/250k_pld_processed.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/opt/data/private/moffusion/outputs/sdf_pore_correlation",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
    )
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--top_k_corr", type=int, default=20)
    return parser.parse_args()


def plain_sdf_names(sdf_dir: str) -> List[str]:
    names = []
    for entry in os.scandir(sdf_dir):
        name = entry.name
        if name.endswith(".npy") and not name.endswith("_occ.npy"):
            names.append(name[:-4])
    return sorted(names)


def load_properties(property_csv: str, targets: List[str]) -> pd.DataFrame:
    usecols = ["name"] + targets
    df = pd.read_csv(property_csv, usecols=usecols)
    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
    return df


def match_names(sdf_dir: str, property_csv: str, targets: List[str]) -> MatchInfo:
    prop_df = load_properties(property_csv, targets)
    matched = []
    for name in prop_df["name"]:
        if os.path.exists(os.path.join(sdf_dir, f"{name}.npy")):
            matched.append(name)
    return MatchInfo(
        matched_names=matched,
        sdf_count=-1,
        property_count=len(prop_df),
    )


def sample_matched_names(
    sdf_dir: str,
    property_csv: str,
    targets: List[str],
    limit: int,
    seed: int,
) -> MatchInfo:
    prop_df = load_properties(property_csv, targets)
    prop_names = set(prop_df["name"])
    matched = []
    for entry in os.scandir(sdf_dir):
        name = entry.name
        if name.endswith(".npy") and not name.endswith("_occ.npy"):
            stem = name[:-4]
            if stem in prop_names:
                matched.append(stem)
                if len(matched) >= limit:
                    break
    return MatchInfo(
        matched_names=matched,
        sdf_count=-1,
        property_count=len(prop_df),
    )


def channel_slices(arr: np.ndarray) -> List[np.ndarray]:
    if arr.ndim == 4 and arr.shape[0] <= 8:
        return [arr[i] for i in range(arr.shape[0])]
    return [arr]


def summarize_vector(vec: np.ndarray, prefix: str) -> Dict[str, float]:
    flat = vec.astype(np.float64, copy=False).reshape(-1)
    qs = np.quantile(flat, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std()),
        f"{prefix}_min": float(flat.min()),
        f"{prefix}_max": float(flat.max()),
        f"{prefix}_q05": float(qs[0]),
        f"{prefix}_q25": float(qs[1]),
        f"{prefix}_q50": float(qs[2]),
        f"{prefix}_q75": float(qs[3]),
        f"{prefix}_q95": float(qs[4]),
        f"{prefix}_pos_frac": float(np.mean(flat > 0)),
        f"{prefix}_neg_frac": float(np.mean(flat < 0)),
    }


def extract_features(name: str, sdf_dir: str) -> Dict[str, float]:
    arr = np.load(os.path.join(sdf_dir, f"{name}.npy"))
    feat: Dict[str, float] = {
        "name": name,
        "ndim": float(arr.ndim),
        "numel": float(arr.size),
    }
    feat.update(summarize_vector(arr, "all"))
    channels = channel_slices(arr)
    feat["num_channels"] = float(len(channels))
    for idx, channel in enumerate(channels):
        feat.update(summarize_vector(channel, f"ch{idx}"))
    return feat


def build_feature_table(names: List[str], sdf_dir: str) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(names, start=1):
        rows.append(extract_features(name, sdf_dir))
        if idx % 1000 == 0:
            print(f"feature_progress {idx}/{len(names)}")
    return pd.DataFrame(rows)


def top_correlations(df: pd.DataFrame, targets: List[str], top_k: int) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in {"name", *targets}]
    rows = []
    for target in targets:
        for feat in feature_cols:
            pearson = df[feat].corr(df[target], method="pearson")
            spearman = df[feat].corr(df[target], method="spearman")
            rows.append(
                {
                    "target": target,
                    "feature": feat,
                    "pearson": float(pearson),
                    "spearman": float(spearman),
                    "abs_pearson": float(abs(pearson)),
                    "abs_spearman": float(abs(spearman)),
                }
            )
    corr_df = pd.DataFrame(rows)
    top_parts = []
    for target in targets:
        part = corr_df[corr_df["target"] == target].sort_values(
            ["abs_spearman", "abs_pearson"], ascending=False
        )
        top_parts.append(part.head(top_k))
    return pd.concat(top_parts, ignore_index=True)


def evaluate_models(df: pd.DataFrame, targets: List[str], seed: int, test_size: float) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in {"name", *targets}]
    X = df[feature_cols]
    rows = []

    ridge = Pipeline(
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
    forest = Pipeline(
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
    models = {
        "ridge": ridge,
        "extra_trees": forest,
    }

    for target in targets:
        y = df[target]
        mask = y.notna()
        X_t = X.loc[mask]
        y_t = y.loc[mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X_t, y_t, test_size=test_size, random_state=seed
        )
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "r2": float(r2_score(y_test, pred)),
                    "mae": float(mean_absolute_error(y_test, pred)),
                    "spearman": float(pd.Series(pred).corr(pd.Series(y_test).reset_index(drop=True), method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    args: argparse.Namespace,
    match: MatchInfo,
    merged_df: pd.DataFrame,
    output_dir: str,
) -> None:
    summary = {
        "sdf_dir": args.sdf_dir,
        "property_csv": args.property_csv,
        "sdf_count": match.sdf_count,
        "property_count": match.property_count,
        "matched_count": len(match.matched_names),
        "used_count": len(merged_df),
        "targets": args.targets,
        "limit": args.limit,
        "seed": args.seed,
        "test_size": args.test_size,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.limit and args.limit > 0:
        match = sample_matched_names(
            args.sdf_dir, args.property_csv, args.targets, args.limit, args.seed
        )
    else:
        match = match_names(args.sdf_dir, args.property_csv, args.targets)
    names = match.matched_names
    print(f"matched_total {len(match.matched_names)}")
    print(f"used_for_run {len(names)}")

    prop_df = load_properties(args.property_csv, args.targets)
    feat_df = build_feature_table(names, args.sdf_dir)
    merged = feat_df.merge(prop_df, on="name", how="inner")

    top_corr = top_correlations(merged, args.targets, args.top_k_corr)
    metrics = evaluate_models(merged, args.targets, args.seed, args.test_size)

    feat_df.to_csv(os.path.join(args.output_dir, "sdf_feature_table.csv"), index=False)
    merged.to_csv(os.path.join(args.output_dir, "merged_sdf_properties.csv"), index=False)
    top_corr.to_csv(os.path.join(args.output_dir, "top_correlations.csv"), index=False)
    metrics.to_csv(os.path.join(args.output_dir, "model_metrics.csv"), index=False)
    write_summary(args, match, merged, args.output_dir)

    print(top_corr.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
